# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _apply_policy_loss_sample_weight(response_mask: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    """Apply per-sample actor-loss weights to a response-token mask.

    This is deliberately independent of veRL's ``multi_turn`` switch. Path-OPD
    flattens environment turns into samples while leaving that switch disabled,
    so routing D12 through ``loss_mask`` would silently leave the actor loss
    unchanged.
    """
    if sample_weight.ndim != 1 or sample_weight.shape[0] != response_mask.shape[0]:
        raise ValueError(
            "policy_loss_sample_weight must have shape (batch_size,), "
            f"got {tuple(sample_weight.shape)} for response mask {tuple(response_mask.shape)}"
        )
    sample_weight = sample_weight.to(dtype=response_mask.dtype, device=response_mask.device).clamp(min=0.0)
    return response_mask * sample_weight.unsqueeze(-1)


def _micro_batch_loss_scale_factor(
    *,
    use_dynamic_bsz: bool,
    micro_batch_size: int,
    mini_batch_size: int,
    gradient_accumulation: Optional[int],
) -> float:
    """Return the same micro-batch weight used for backward and loss metrics."""
    if use_dynamic_bsz:
        return micro_batch_size / mini_batch_size
    if not gradient_accumulation or gradient_accumulation <= 0:
        raise ValueError(f"gradient_accumulation must be positive, got {gradient_accumulation}")
    return 1.0 / gradient_accumulation


def _accumulate_scaled_loss_metric(
    metric_sums: Dict[str, float], key: str, value: Union[torch.Tensor, float], loss_scale_factor: float
) -> None:
    """Accumulate a micro-batch loss exactly as it contributes to backward."""
    if isinstance(value, torch.Tensor):
        value = value.detach().item()
    metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * loss_scale_factor


def _accumulate_ratio_metric(
    metric_parts: Dict[str, Tuple[float, float]],
    key: str,
    numerator: Union[torch.Tensor, float],
    denominator: Union[torch.Tensor, float],
) -> None:
    """Accumulate ratio numerators and denominators across micro-batches."""
    if isinstance(numerator, torch.Tensor):
        numerator = numerator.detach().item()
    if isinstance(denominator, torch.Tensor):
        denominator = denominator.detach().item()
    old_numerator, old_denominator = metric_parts.get(key, (0.0, 0.0))
    metric_parts[key] = (old_numerator + float(numerator), old_denominator + float(denominator))


def _finalize_ratio_metrics(metric_parts: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    return {
        key: numerator / max(denominator, 1e-8)
        for key, (numerator, denominator) in metric_parts.items()
    }


def _extract_teacher_topk_log_probs(
    logits: torch.Tensor,
    row_indices: torch.Tensor,
    topk: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract Top-k full-vocabulary log-probs without a full fp32 copy."""
    if row_indices.numel() == 0:
        return (
            torch.empty((0, topk), dtype=torch.int32, device=logits.device),
            torch.empty((0, topk), dtype=torch.float32, device=logits.device),
        )
    topk_ids_chunks = []
    topk_log_prob_chunks = []
    for start in range(0, row_indices.numel(), chunk_size):
        rows = row_indices[start : start + chunk_size]
        chunk_logits = logits.index_select(0, rows)
        topk_logits, topk_ids = torch.topk(chunk_logits, k=topk, dim=-1)
        log_z = torch.logsumexp(chunk_logits.float(), dim=-1, keepdim=True)
        topk_ids_chunks.append(topk_ids.to(torch.int32))
        topk_log_prob_chunks.append(topk_logits.float() - log_z)
    return torch.cat(topk_ids_chunks, dim=0), torch.cat(topk_log_prob_chunks, dim=0)


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        return_topk=0,
        teacher_topk_ids=None,
        teacher_topk_log_probs=None,
        topk_include_tail=True,
    ):
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            auxiliary_outputs: optional Top-k tensors, each response-aligned
        """
        response_length = micro_batch["responses"].size(-1)
        topk_chunk_size = int(self.config.get("sdl_topk_chunk_size", 256))
        if topk_chunk_size <= 0:
            raise ValueError(f"sdl_topk_chunk_size must be positive, got {topk_chunk_size}")
        return_topk = int(return_topk or 0)
        compute_topk_loss = teacher_topk_ids is not None or teacher_topk_log_probs is not None
        needs_topk_alignment = bool(return_topk or compute_topk_loss)
        if (teacher_topk_ids is None) != (teacher_topk_log_probs is None):
            raise ValueError("teacher_topk_ids and teacher_topk_log_probs must be provided together")
        if return_topk < 0:
            raise ValueError(f"return_topk must be non-negative, got {return_topk}")
        if return_topk and compute_topk_loss:
            raise ValueError("A forward pass cannot both produce teacher Top-k and consume a Top-k target")
        if needs_topk_alignment and self.use_fused_kernels:
            raise ValueError("Top-k SDL requires unfused model logits; set actor.use_fused_kernels=False")

        auxiliary_outputs = {}
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                if needs_topk_alignment:
                    # Map every unpadded model-output row back to a response-token
                    # prediction position. Logit position ``seqlen-R-1+j`` predicts
                    # response token ``j``. Chosen-token/ref forwards do not need
                    # these full-packed-sequence tensors and retain the pre-Top-k
                    # memory path.
                    flat_indices = indices.long()
                    flat_positions = flat_indices.remainder(seqlen)
                    response_positions_rmpad = flat_positions - (seqlen - response_length - 1)
                    response_batches_rmpad = torch.div(flat_indices, seqlen, rounding_mode="floor")
                    response_prediction_mask_rmpad = (response_positions_rmpad >= 0) & (response_positions_rmpad < response_length)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                    if needs_topk_alignment:
                        if is_vlm_model:
                            raise NotImplementedError("Top-k SDL with multimodal Ulysses sequence parallelism is not supported")
                        if pad_size:
                            response_positions_rmpad = torch.cat(
                                [response_positions_rmpad, response_positions_rmpad.new_full((pad_size,), -1)]
                            )
                            response_batches_rmpad = torch.cat(
                                [response_batches_rmpad, response_batches_rmpad.new_full((pad_size,), -1)]
                            )
                            response_prediction_mask_rmpad = torch.cat(
                                [response_prediction_mask_rmpad, response_prediction_mask_rmpad.new_zeros((pad_size,))]
                            )
                        response_positions_rmpad = slice_input_tensor(response_positions_rmpad, dim=0, padding=False)
                        response_batches_rmpad = slice_input_tensor(response_batches_rmpad, dim=0, padding=False)
                        response_prediction_mask_rmpad = slice_input_tensor(
                            response_prediction_mask_rmpad, dim=0, padding=False
                        )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    if calculate_entropy:
                        entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy or compute_topk_loss:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                    if return_topk:
                        if return_topk > logits_rmpad.shape[-1]:
                            raise ValueError(f"Top-k={return_topk} exceeds vocabulary size {logits_rmpad.shape[-1]}")
                        valid_rows = response_prediction_mask_rmpad.nonzero(as_tuple=False).squeeze(-1)
                        topk_ids, topk_log_probs = _extract_teacher_topk_log_probs(
                            logits=logits_rmpad,
                            row_indices=valid_rows,
                            topk=return_topk,
                            chunk_size=topk_chunk_size,
                        )
                        teacher_ids_rmpad = torch.zeros(
                            (logits_rmpad.shape[0], return_topk), dtype=torch.int32, device=logits_rmpad.device
                        ).index_copy(0, valid_rows, topk_ids)
                        teacher_log_probs_rmpad = torch.zeros(
                            (logits_rmpad.shape[0], return_topk), dtype=torch.float32, device=logits_rmpad.device
                        ).index_copy(0, valid_rows, topk_log_probs)

                    if compute_topk_loss:
                        from verl.trainer.ppo.skillsd_utils import compute_topk_forward_kl_with_tail

                        valid_batches = response_batches_rmpad[response_prediction_mask_rmpad]
                        valid_positions = response_positions_rmpad[response_prediction_mask_rmpad]
                        valid_teacher_ids = teacher_topk_ids[valid_batches, valid_positions]
                        valid_teacher_log_probs = teacher_topk_log_probs[valid_batches, valid_positions]
                        valid_rows = response_prediction_mask_rmpad.nonzero(as_tuple=False).squeeze(-1)
                        if valid_rows.numel() == 0:
                            # Keep a zero-valued dependency on local logits so
                            # every sequence-parallel rank participates in the
                            # gather backward collective.
                            topk_loss = logits_rmpad.sum(dim=-1)[:0].float()
                            topk_diagnostics = {
                                name: torch.empty(0, dtype=torch.float32, device=logits_rmpad.device)
                                for name in ("teacher_mass", "student_mass", "teacher_tail_mass", "student_tail_mass")
                            }
                        else:
                            loss_chunks = []
                            diagnostic_chunks = {}
                            for start in range(0, valid_rows.numel(), topk_chunk_size):
                                end = min(start + topk_chunk_size, valid_rows.numel())
                                loss_chunk, diagnostics_chunk = compute_topk_forward_kl_with_tail(
                                    student_logits=logits_rmpad.index_select(0, valid_rows[start:end]),
                                    teacher_topk_ids=valid_teacher_ids[start:end],
                                    teacher_topk_log_probs=valid_teacher_log_probs[start:end],
                                    include_tail=bool(topk_include_tail),
                                    chunk_size=topk_chunk_size,
                                )
                                loss_chunks.append(loss_chunk)
                                for name, values in diagnostics_chunk.items():
                                    diagnostic_chunks.setdefault(name, []).append(values)
                            topk_loss = torch.cat(loss_chunks, dim=0)
                            topk_diagnostics = {
                                name: torch.cat(values, dim=0) for name, values in diagnostic_chunks.items()
                            }
                        auxiliary_rmpad = {
                            "topk_sdl_per_token": torch.zeros(
                                logits_rmpad.shape[0], dtype=topk_loss.dtype, device=logits_rmpad.device
                            ).index_copy(0, valid_rows, topk_loss)
                        }
                        for name, values in topk_diagnostics.items():
                            auxiliary_rmpad[f"topk_{name}"] = torch.zeros(
                                logits_rmpad.shape[0], dtype=values.dtype, device=logits_rmpad.device
                            ).index_copy(0, valid_rows, values)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if return_topk:
                        teacher_ids_rmpad = gather_outpus_and_unpad(
                            teacher_ids_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size, grad_scaler=False
                        )
                        teacher_log_probs_rmpad = gather_outpus_and_unpad(
                            teacher_log_probs_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size, grad_scaler=False
                        )
                    if compute_topk_loss:
                        auxiliary_rmpad = {
                            name: gather_outpus_and_unpad(
                                values, gather_dim=0, unpad_dim=0, padding_size=pad_size, grad_scaler=name == "topk_sdl_per_token"
                            )
                            for name, values in auxiliary_rmpad.items()
                        }
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if return_topk:
                    full_teacher_ids = pad_input(teacher_ids_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
                    full_teacher_log_probs = pad_input(
                        teacher_log_probs_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                    )
                    auxiliary_outputs["teacher_topk_ids"] = full_teacher_ids[:, -response_length - 1 : -1]
                    auxiliary_outputs["teacher_topk_log_probs"] = full_teacher_log_probs[:, -response_length - 1 : -1]
                if compute_topk_loss:
                    for name, values in auxiliary_rmpad.items():
                        full_values = pad_input(values.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                        auxiliary_outputs[name] = full_values.squeeze(-1)[:, -response_length - 1 : -1]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    if calculate_entropy:
                        entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(
                        logits,
                        micro_batch["responses"],
                        inplace_backward=not (calculate_entropy or compute_topk_loss),
                    )
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                    if return_topk:
                        if return_topk > logits.shape[-1]:
                            raise ValueError(f"Top-k={return_topk} exceeds vocabulary size {logits.shape[-1]}")
                        flat_logits = logits.reshape(-1, logits.shape[-1])
                        flat_rows = torch.arange(flat_logits.shape[0], device=flat_logits.device)
                        topk_ids, topk_log_probs = _extract_teacher_topk_log_probs(
                            logits=flat_logits,
                            row_indices=flat_rows,
                            topk=return_topk,
                            chunk_size=topk_chunk_size,
                        )
                        auxiliary_outputs["teacher_topk_ids"] = topk_ids.reshape(*logits.shape[:-1], return_topk)
                        auxiliary_outputs["teacher_topk_log_probs"] = topk_log_probs.reshape(
                            *logits.shape[:-1], return_topk
                        )

                    if compute_topk_loss:
                        from verl.trainer.ppo.skillsd_utils import compute_topk_forward_kl_with_tail

                        topk_loss, topk_diagnostics = compute_topk_forward_kl_with_tail(
                            student_logits=logits,
                            teacher_topk_ids=teacher_topk_ids,
                            teacher_topk_log_probs=teacher_topk_log_probs,
                            include_tail=bool(topk_include_tail),
                            chunk_size=topk_chunk_size,
                        )
                        auxiliary_outputs["topk_sdl_per_token"] = topk_loss
                        auxiliary_outputs.update({f"topk_{name}": values for name, values in topk_diagnostics.items()})

            return entropy, log_probs, auxiliary_outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        return_topk = int(data.meta_info.get("return_topk", 0) or 0)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        auxiliary_output_lists = {}
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, auxiliary_outputs = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    return_topk=return_topk,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            for name, values in auxiliary_outputs.items():
                auxiliary_output_lists.setdefault(name, []).append(values)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        auxiliary_outputs = {name: torch.concat(values, dim=0) for name, values in auxiliary_output_lists.items()}
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            auxiliary_outputs = {name: values[revert_indices] for name, values in auxiliary_outputs.items()}

        return log_probs, entropys, auxiliary_outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if "policy_loss_sample_weight" in data.batch:
            select_keys.append("policy_loss_sample_weight")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        use_sdl_loss = bool(self.config.get("use_sdl_loss", False))
        use_sdar_loss = bool(self.config.get("use_sdar_loss", False))
        use_candidate_ce_loss = bool(self.config.get("use_candidate_ce_loss", False))
        if use_candidate_ce_loss and (use_sdl_loss or use_sdar_loss):
            raise ValueError(
                "candidate CE is a standalone auxiliary-loss control and cannot be combined "
                "with SDL or SDAR in the same actor update"
            )
        sdl_loss_mode = str(self.config.get("sdl_loss_mode", "chosen_token_k3") or "chosen_token_k3").lower()
        topk_sdl_modes = ("topk", "topk_forward_kl", "teacher_topk_forward_kl", "forward_kl_topk")
        use_topk_sdl = use_sdl_loss and sdl_loss_mode in topk_sdl_modes
        if (use_sdl_loss and not use_topk_sdl) or use_sdar_loss:
            select_keys.append("teacher_log_probs")
        if use_topk_sdl:
            select_keys.extend(["teacher_topk_ids", "teacher_topk_log_probs"])
        if use_candidate_ce_loss:
            select_keys.extend(
                [
                    "candidate_ce_input_ids",
                    "candidate_ce_attention_mask",
                    "candidate_ce_position_ids",
                    "candidate_ce_responses",
                    "candidate_ce_response_mask",
                    "candidate_ce_sample_weight",
                    "candidate_ce_normalization_token_count",
                ]
            )
        sdl_sample_filter = str(self.config.get("sdl_loss_sample_filter", "all") or "all").lower()
        sdl_token_scope = str(self.config.get("sdl_loss_token_scope", "all") or "all").lower()
        sdl_mask_special_tokens = bool(self.config.get("sdl_loss_mask_special_tokens", False))
        sdar_mask_special_tokens = bool(self.config.get("sdar_loss_mask_special_tokens", False))
        sdl_sample_weighting = bool(self.config.get("sdl_loss_sample_weighting", False))
        sdl_loss_normalization = str(self.config.get("sdl_loss_normalization", "selected_token_mean") or "selected_token_mean").lower()
        if self.config.get("use_sdl_loss", False) and sdl_sample_filter not in ("all", "none", ""):
            select_keys.append("token_level_scores")
        if self.config.get("use_sdl_loss", False) and sdl_token_scope in ("action", "action_only", "action_span"):
            select_keys.append("sdl_action_mask")
        if self.config.get("use_sdl_loss", False) and sdl_mask_special_tokens:
            select_keys.append("sdl_special_token_keep_mask")
        if use_sdar_loss and sdar_mask_special_tokens:
            select_keys.append("sdl_special_token_keep_mask")
        if self.config.get("use_sdl_loss", False) and sdl_sample_weighting:
            select_keys.append("sdl_sample_weight")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                scaled_loss_metrics = {}
                sdl_ratio_metric_parts = {}

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]
                    loss_scale_factor = _micro_batch_loss_scale_factor(
                        use_dynamic_bsz=self.config.use_dynamic_bsz,
                        micro_batch_size=len(data),
                        mini_batch_size=self.config.ppo_mini_batch_size,
                        gradient_accumulation=getattr(self, "gradient_accumulation", None),
                    )
                    if "policy_loss_sample_weight" in data:
                        base_response_token_count = response_mask.float().sum().clamp(min=1.0)
                        policy_loss_sample_weight = data["policy_loss_sample_weight"]
                        response_mask = _apply_policy_loss_sample_weight(response_mask, policy_loss_sample_weight)
                        metrics["actor/policy_loss_sample_weight_mean"] = policy_loss_sample_weight.float().clamp(min=0.0).mean().detach().item()
                        metrics["actor/policy_loss_weighted_token_ratio"] = (response_mask.float().sum() / base_response_token_count).detach().item()

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob, forward_auxiliary_outputs = self._forward_micro_batch(
                        micro_batch=data,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        teacher_topk_ids=data.get("teacher_topk_ids") if use_topk_sdl else None,
                        teacher_topk_log_probs=data.get("teacher_topk_log_probs") if use_topk_sdl else None,
                        topk_include_tail=bool(self.config.get("sdl_topk_include_tail", True)),
                    )
                    
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    pg_loss_coef = self.config.get("pg_loss_coef", 1.0)
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss * pg_loss_coef - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss * pg_loss_coef

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        _accumulate_scaled_loss_metric(
                            scaled_loss_metrics, "actor/kl_loss", kl_loss, loss_scale_factor
                        )
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if use_candidate_ce_loss:
                        from verl.trainer.ppo.skillsd_utils import compute_candidate_ce_loss

                        # Backpropagate the GRPO graph before constructing the
                        # extra candidate graph. Gradients still add within the
                        # same optimizer step, while peak activation memory does
                        # not include both prompt forwards at once.
                        (policy_loss * loss_scale_factor).backward()
                        candidate_target_mask = data["candidate_ce_response_mask"]
                        candidate_sample_weight = data["candidate_ce_sample_weight"].float().clamp(min=0.0)
                        candidate_weighted_mask = (
                            candidate_target_mask.float() * candidate_sample_weight.unsqueeze(-1)
                        )
                        candidate_normalization_count = data[
                            "candidate_ce_normalization_token_count"
                        ].float()
                        _accumulate_ratio_metric(
                            sdl_ratio_metric_parts,
                            "actor/candidate_ce_selected_sample_ratio",
                            (candidate_sample_weight > 0).float().sum(),
                            len(candidate_sample_weight),
                        )
                        _accumulate_ratio_metric(
                            sdl_ratio_metric_parts,
                            "actor/candidate_ce_target_to_response_token_ratio",
                            candidate_weighted_mask.sum(),
                            candidate_normalization_count.sum().clamp(min=1.0),
                        )
                        # Every FSDP rank must execute the same candidate
                        # forward/backward collectives.  Match-only selection
                        # can leave a local micro-batch with zero selected
                        # rows even when another rank has selected rows.
                        # Skipping the graph locally would desynchronize the
                        # process groups and eventually trip the NCCL
                        # watchdog.  Zero sample weights already make this a
                        # differentiable zero loss on unselected ranks.
                        candidate_micro_batch = {
                            "input_ids": data["candidate_ce_input_ids"],
                            "attention_mask": data["candidate_ce_attention_mask"],
                            "position_ids": data["candidate_ce_position_ids"],
                            "responses": data["candidate_ce_responses"],
                        }
                        _, candidate_log_prob, _ = self._forward_micro_batch(
                            micro_batch=candidate_micro_batch,
                            temperature=1.0,
                            calculate_entropy=False,
                        )
                        candidate_ce_loss = compute_candidate_ce_loss(
                            candidate_log_probs=candidate_log_prob,
                            target_mask=candidate_target_mask,
                            sample_weight=candidate_sample_weight,
                            normalization_token_count=candidate_normalization_count,
                            loss_normalization=self.config.get(
                                "candidate_ce_loss_normalization",
                                "response_token_mean",
                            ),
                        )
                        candidate_ce_coef = self.config.get("candidate_ce_loss_coef", 0.01)
                        (
                            candidate_ce_loss
                            * candidate_ce_coef
                            * loss_scale_factor
                        ).backward()
                        _accumulate_scaled_loss_metric(
                            scaled_loss_metrics,
                            "actor/candidate_ce_loss",
                            candidate_ce_loss,
                            loss_scale_factor,
                        )
                        append_to_dict(
                            metrics,
                            {
                                "actor/candidate_ce_loss_micro_max": candidate_ce_loss.detach().item()
                            },
                        )
                        metrics["actor/candidate_ce_coef"] = candidate_ce_coef
                        metrics["actor/candidate_ce_no_teacher_forward"] = 1.0
                        metrics[
                            "actor/candidate_ce_normalization_response_token_mean"
                        ] = float(
                            str(
                                self.config.get(
                                    "candidate_ce_loss_normalization",
                                    "response_token_mean",
                                )
                            ).lower()
                            in (
                                "response",
                                "response_token",
                                "response_token_mean",
                                "original_response_token_mean",
                            )
                        )

                    if use_sdl_loss:
                        from verl.trainer.ppo.skillsd_utils import aggregate_sdl_per_token_loss, compute_sdl_loss

                        sdl_response_mask = response_mask
                        response_token_count = response_mask.float().sum().clamp(min=1.0)
                        if sdl_mask_special_tokens:
                            if "sdl_special_token_keep_mask" not in data:
                                raise KeyError("sdl_loss_mask_special_tokens=True requires batch key 'sdl_special_token_keep_mask'")
                            special_keep_mask = data["sdl_special_token_keep_mask"][:, -response_length:].to(dtype=response_mask.dtype)
                            sdl_response_mask = sdl_response_mask * special_keep_mask
                            _accumulate_ratio_metric(
                                sdl_ratio_metric_parts,
                                "actor/sdl_non_special_token_ratio",
                                sdl_response_mask.float().sum(),
                                response_token_count,
                            )
                            metrics["actor/sdl_special_token_mask_enabled"] = 1.0
                        if sdl_sample_filter not in ("all", "none", ""):
                            if sdl_sample_filter in ("failed", "failure", "failed_only", "failure_only", "non_success"):
                                sample_scores = data["token_level_scores"].sum(dim=-1)
                                sample_mask = sample_scores <= 0
                            elif sdl_sample_filter in ("success", "successful", "success_only"):
                                sample_scores = data["token_level_scores"].sum(dim=-1)
                                sample_mask = sample_scores > 0
                            else:
                                raise ValueError(f"Unsupported sdl_loss_sample_filter={sdl_sample_filter!r}; expected all, failed_only, or success_only")
                            sdl_response_mask = sdl_response_mask * sample_mask.unsqueeze(-1).to(dtype=response_mask.dtype)
                            _accumulate_ratio_metric(
                                sdl_ratio_metric_parts,
                                "actor/sdl_filter_sample_ratio",
                                sample_mask.float().sum(),
                                len(sample_mask),
                            )
                            _accumulate_ratio_metric(
                                sdl_ratio_metric_parts,
                                "actor/sdl_filter_token_ratio",
                                sdl_response_mask.float().sum(),
                                response_token_count,
                            )

                        if sdl_token_scope in ("action", "action_only", "action_span"):
                            if "sdl_action_mask" not in data:
                                raise KeyError("sdl_loss_token_scope=action_only requires batch key 'sdl_action_mask'")
                            action_mask = data["sdl_action_mask"][:, -response_length:].to(dtype=response_mask.dtype)
                            sdl_response_mask = sdl_response_mask * action_mask
                            _accumulate_ratio_metric(
                                sdl_ratio_metric_parts,
                                "actor/sdl_action_token_ratio",
                                action_mask.float().sum(),
                                response_token_count,
                            )
                            _accumulate_ratio_metric(
                                sdl_ratio_metric_parts,
                                "actor/sdl_effective_token_ratio",
                                sdl_response_mask.float().sum(),
                                response_token_count,
                            )
                            metrics["actor/sdl_token_scope_action_only"] = 1.0

                        if sdl_sample_weighting:
                            if "sdl_sample_weight" not in data:
                                raise KeyError("sdl_loss_sample_weighting=True requires batch key 'sdl_sample_weight'")
                            sample_weight = data["sdl_sample_weight"].to(dtype=response_mask.dtype, device=response_mask.device).clamp(min=0.0)
                            sdl_response_mask = sdl_response_mask * sample_weight.unsqueeze(-1)
                            _accumulate_ratio_metric(
                                sdl_ratio_metric_parts,
                                "actor/sdl_sample_weight_mean",
                                sample_weight.float().sum(),
                                len(sample_weight),
                            )
                            _accumulate_ratio_metric(
                                sdl_ratio_metric_parts,
                                "actor/sdl_weighted_token_ratio",
                                sdl_response_mask.float().sum(),
                                response_token_count,
                            )

                        if sdl_response_mask.sum().item() <= 0:
                            sdl_loss = log_prob.sum() * 0.0
                        elif use_topk_sdl:
                            from verl.trainer.ppo.skillsd_utils import stabilize_topk_sdl_per_token_loss

                            if "topk_sdl_per_token" not in forward_auxiliary_outputs:
                                raise KeyError("Top-k SDL forward pass did not return topk_sdl_per_token")
                            topk_use_is_weight = bool(self.config.get("sdl_topk_is_weight", False))
                            topk_loss_clamp = float(self.config.get("sdl_topk_loss_clamp", 0.0) or 0.0)
                            topk_per_token, topk_stability_diagnostics = stabilize_topk_sdl_per_token_loss(
                                forward_auxiliary_outputs["topk_sdl_per_token"],
                                log_prob=log_prob,
                                old_log_prob=old_log_prob,
                                use_is_weight=topk_use_is_weight,
                                loss_clamp=topk_loss_clamp,
                            )
                            sdl_loss = aggregate_sdl_per_token_loss(
                                per_token_loss=topk_per_token,
                                response_mask=sdl_response_mask,
                                loss_agg_mode=loss_agg_mode,
                                normalization_mask=response_mask,
                                loss_normalization=sdl_loss_normalization,
                            )
                            diagnostic_denom = sdl_response_mask.float().sum().clamp(min=1.0)
                            if topk_loss_clamp > 0.0:
                                _accumulate_ratio_metric(
                                    sdl_ratio_metric_parts,
                                    "actor/sdl_topk_clamped_token_ratio",
                                    (topk_stability_diagnostics["clamped_mask"] * sdl_response_mask.float()).sum(),
                                    diagnostic_denom,
                                )
                                metrics["actor/sdl_topk_loss_clamp"] = topk_loss_clamp
                            if topk_use_is_weight:
                                _accumulate_ratio_metric(
                                    sdl_ratio_metric_parts,
                                    "actor/sdl_topk_is_weight_mean",
                                    (topk_stability_diagnostics["is_weight"].float() * sdl_response_mask.float()).sum(),
                                    diagnostic_denom,
                                )
                                metrics["actor/sdl_topk_is_weight_enabled"] = 1.0
                            diagnostic_names = (
                                "teacher_mass",
                                "student_mass",
                                "teacher_tail_mass",
                                "student_tail_mass",
                            )
                            for name in diagnostic_names:
                                values = forward_auxiliary_outputs[f"topk_{name}"].detach().float()
                                _accumulate_ratio_metric(
                                    sdl_ratio_metric_parts,
                                    f"actor/sdl_topk_{name}",
                                    (values * sdl_response_mask.float()).sum(),
                                    diagnostic_denom,
                                )
                            metrics["actor/sdl_topk_k"] = float(data["teacher_topk_ids"].shape[-1])
                            metrics["actor/sdl_topk_include_tail"] = float(
                                bool(self.config.get("sdl_topk_include_tail", True))
                            )
                            metrics["actor/sdl_loss_mode_topk_forward_kl"] = 1.0
                        else:
                            teacher_log_probs = data["teacher_log_probs"]
                            sdl_loss = compute_sdl_loss(
                                student_log_probs=log_prob,
                                teacher_log_probs=teacher_log_probs,
                                old_log_probs=old_log_prob,
                                response_mask=sdl_response_mask,
                                loss_agg_mode=loss_agg_mode,
                                normalization_mask=response_mask,
                                loss_normalization=sdl_loss_normalization,
                            )
                        sdl_coef = self.config.get("sdl_loss_coef", 0.1)
                        policy_loss = policy_loss + sdl_loss * sdl_coef
                        _accumulate_scaled_loss_metric(
                            scaled_loss_metrics, "actor/sdl_loss", sdl_loss, loss_scale_factor
                        )
                        append_to_dict(metrics, {"actor/sdl_loss_micro_max": sdl_loss.detach().item()})
                        metrics["actor/sdl_coef"] = sdl_coef
                        metrics["actor/sdl_loss_normalization_response_token_mean"] = float(
                            sdl_loss_normalization in ("response", "response_token", "response_token_mean", "original_response_token_mean")
                        )

                    if use_sdar_loss:
                        from verl.trainer.ppo.sdar_utils import compute_sdar_loss
                        teacher_log_probs = data["teacher_log_probs"]
                        sdar_response_mask = response_mask
                        if sdar_mask_special_tokens:
                            if "sdl_special_token_keep_mask" not in data:
                                raise KeyError("sdar_loss_mask_special_tokens=True requires batch key 'sdl_special_token_keep_mask'")
                            special_keep_mask = data["sdl_special_token_keep_mask"][:, -response_length:].to(dtype=response_mask.dtype)
                            sdar_response_mask = sdar_response_mask * special_keep_mask
                            response_token_count = response_mask.float().sum().clamp(min=1.0)
                            metrics["sdar/non_special_token_ratio"] = (sdar_response_mask.float().sum() / response_token_count).detach().item()
                            metrics["sdar/special_token_mask_enabled"] = 1.0
                        sdar_loss, sdar_metrics = compute_sdar_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_probs,
                            response_mask=sdar_response_mask,
                            gate_beta=self.config.get("sdar_gate_beta", 5.0),
                            loss_agg_mode=loss_agg_mode,
                        )
                        sdar_coef = self.config.get("sdar_loss_coef", 0.1)
                        policy_loss = policy_loss + sdar_loss * sdar_coef
                        metrics.update(sdar_metrics)
                        metrics["sdar/coef"] = sdar_coef

                    # Use exactly the same scale for backward and reported loss
                    # metrics. This is the standard veRL dynamic-batch weighting:
                    # each micro-batch contributes in proportion to its sample
                    # count within the PPO mini-batch.
                    if not use_candidate_ce_loss:
                        loss = policy_loss * loss_scale_factor
                        loss.backward()

                    _accumulate_scaled_loss_metric(
                        scaled_loss_metrics, "actor/pg_loss", pg_loss, loss_scale_factor
                    )
                    data = {
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                append_to_dict(metrics, scaled_loss_metrics)
                append_to_dict(metrics, _finalize_ratio_metrics(sdl_ratio_metric_parts))
                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
