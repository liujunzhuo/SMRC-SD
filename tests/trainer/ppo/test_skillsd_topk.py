# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from verl.trainer.ppo.skillsd_utils import (
    aggregate_sdl_per_token_loss,
    compute_candidate_ce_loss,
    compute_sdl_loss,
    compute_topk_forward_kl_with_tail,
    stabilize_topk_sdl_per_token_loss,
)
from verl.workers.actor.dp_actor import (
    DataParallelPPOActor,
    _accumulate_scaled_loss_metric,
    _accumulate_ratio_metric,
    _finalize_ratio_metrics,
    _micro_batch_loss_scale_factor,
)


class _TokenTableModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        values = torch.arange(vocab_size * vocab_size, dtype=torch.float32).reshape(vocab_size, vocab_size) / 10
        self.logit_table = nn.Parameter(values)

    def forward(self, input_ids, **_kwargs):
        return type("ModelOutput", (), {"logits": self.logit_table[input_ids]})()


def _coarse_grained_full_kl(student_logits, teacher_logits, teacher_ids):
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    teacher_selected_log_probs = teacher_log_probs.gather(-1, teacher_ids)
    student_selected_log_probs = student_log_probs.gather(-1, teacher_ids)
    teacher_selected_probs = teacher_selected_log_probs.exp()
    student_selected_probs = student_selected_log_probs.exp()
    selected = (teacher_selected_probs * (teacher_selected_log_probs - student_selected_log_probs)).sum(-1)
    teacher_tail = 1.0 - teacher_selected_probs.sum(-1)
    student_tail = 1.0 - student_selected_probs.sum(-1)
    return selected + teacher_tail * (teacher_tail.log() - student_tail.log())


def test_topk_forward_kl_with_tail_matches_coarse_grained_full_distribution():
    teacher_logits = torch.tensor([[2.0, 0.5, -0.7, 1.1, -1.2], [0.2, 1.7, -0.4, 0.8, 0.1]])
    student_logits = torch.tensor([[0.8, 1.2, -0.3, 0.2, -0.4], [1.0, 0.1, -0.2, 1.1, -0.8]], requires_grad=True)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    teacher_topk_log_probs, teacher_topk_ids = teacher_log_probs.topk(k=2, dim=-1)

    actual, diagnostics = compute_topk_forward_kl_with_tail(
        student_logits,
        teacher_topk_ids,
        teacher_topk_log_probs,
    )
    expected = _coarse_grained_full_kl(student_logits, teacher_logits, teacher_topk_ids)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.all(actual >= -1e-7)
    torch.testing.assert_close(
        diagnostics["teacher_mass"] + diagnostics["teacher_tail_mass"],
        torch.ones_like(actual),
    )
    torch.testing.assert_close(
        diagnostics["student_mass"] + diagnostics["student_tail_mass"],
        torch.ones_like(actual),
    )


def test_topk_forward_kl_increases_teacher_preferred_token():
    student_logits = torch.zeros(1, 4, requires_grad=True)
    teacher_logits = torch.tensor([[3.0, 0.0, -1.0, -2.0]])
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    teacher_topk_log_probs, teacher_topk_ids = teacher_log_probs.topk(k=2, dim=-1)

    loss, _ = compute_topk_forward_kl_with_tail(student_logits, teacher_topk_ids, teacher_topk_log_probs)
    loss.sum().backward()

    preferred_id = teacher_topk_ids[0, 0]
    assert student_logits.grad[0, preferred_id] < 0, "gradient descent must raise the teacher's top token logit"
    assert student_logits.grad.sum().abs() < 1e-6


def test_topk_forward_kl_is_zero_when_student_matches_teacher_for_any_k():
    logits = torch.tensor([[1.3, -0.2, 0.7, -1.4, 0.1]], requires_grad=True)
    log_probs = F.log_softmax(logits.detach(), dim=-1)
    for k in (1, 2, 5):
        teacher_topk_log_probs, teacher_topk_ids = log_probs.topk(k=k, dim=-1)
        loss, _ = compute_topk_forward_kl_with_tail(logits, teacher_topk_ids, teacher_topk_log_probs)
        torch.testing.assert_close(loss, torch.zeros_like(loss), atol=2e-7, rtol=0)


def test_topk_sdl_response_normalization_keeps_sparse_scale_explicit():
    per_token_loss = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
    selected_mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    response_mask = torch.ones_like(selected_mask)

    selected_mean = aggregate_sdl_per_token_loss(per_token_loss, selected_mask)
    response_mean = aggregate_sdl_per_token_loss(
        per_token_loss,
        selected_mask,
        normalization_mask=response_mask,
        loss_normalization="response_token_mean",
    )

    torch.testing.assert_close(selected_mean, torch.tensor(4.0))
    torch.testing.assert_close(response_mean, torch.tensor(2.0))


def test_chosen_token_k3_path_keeps_legacy_formula():
    student = torch.tensor([[-1.0, -0.7]], requires_grad=True)
    teacher = torch.tensor([[-0.5, -1.1]])
    old = torch.tensor([[-1.2, -0.8]])
    mask = torch.ones_like(student)
    ell = student - teacher
    expected = (torch.exp(student - old) * (torch.exp(-ell) - 1.0 + ell)).mean()

    actual = compute_sdl_loss(student, teacher, old, mask)

    torch.testing.assert_close(actual, expected)


def test_candidate_ce_response_normalization_matches_sparse_auxiliary_budget():
    candidate_log_probs = torch.tensor(
        [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
        requires_grad=True,
    )
    target_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    sample_weight = torch.tensor([1.0, 0.0])
    sampled_response_token_count = torch.tensor([4.0, 6.0])

    response_mean = compute_candidate_ce_loss(
        candidate_log_probs,
        target_mask,
        sample_weight,
        sampled_response_token_count,
        loss_normalization="response_token_mean",
    )
    selected_mean = compute_candidate_ce_loss(
        candidate_log_probs,
        target_mask,
        sample_weight,
        sampled_response_token_count,
        loss_normalization="selected_token_mean",
    )

    torch.testing.assert_close(response_mean, torch.tensor(0.3))
    torch.testing.assert_close(selected_mean, torch.tensor(1.5))
    response_mean.backward()
    torch.testing.assert_close(
        candidate_log_probs.grad,
        torch.tensor([[-0.1, -0.1, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_candidate_ce_zero_local_selection_retains_zero_gradient_graph():
    candidate_log_probs = torch.tensor(
        [[-1.0, -2.0], [-3.0, -4.0]],
        requires_grad=True,
    )
    loss = compute_candidate_ce_loss(
        candidate_log_probs,
        target_mask=torch.ones_like(candidate_log_probs),
        sample_weight=torch.zeros(2),
        normalization_token_count=torch.tensor([3.0, 5.0]),
        loss_normalization="response_token_mean",
    )

    assert loss.requires_grad
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    torch.testing.assert_close(
        candidate_log_probs.grad,
        torch.zeros_like(candidate_log_probs),
    )


def test_topk_stabilizers_default_off_is_identity():
    per_token = torch.tensor([[0.5, 25.0, 3.0]], requires_grad=True)
    loss, diagnostics = stabilize_topk_sdl_per_token_loss(per_token)

    assert loss is per_token
    assert diagnostics == {}


def test_topk_loss_clamp_caps_tokens_and_reports_mask():
    per_token = torch.tensor([[0.5, 25.0, 3.0]])
    loss, diagnostics = stabilize_topk_sdl_per_token_loss(per_token, loss_clamp=20.0)

    torch.testing.assert_close(loss, torch.tensor([[0.5, 20.0, 3.0]]))
    torch.testing.assert_close(diagnostics["clamped_mask"], torch.tensor([[0.0, 1.0, 0.0]]))


def test_topk_is_weight_is_detached_and_truncated():
    per_token = torch.tensor([[1.0, 1.0, 1.0]])
    log_prob = torch.tensor([[0.0, -1.0, 30.0]], requires_grad=True)
    old_log_prob = torch.tensor([[0.0, 0.0, 0.0]])

    loss, diagnostics = stabilize_topk_sdl_per_token_loss(
        per_token,
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        use_is_weight=True,
        is_log_ratio_clamp=10.0,
    )

    expected = torch.tensor([[1.0, torch.exp(torch.tensor(-1.0)).item(), torch.exp(torch.tensor(10.0)).item()]])
    torch.testing.assert_close(loss, expected)
    assert not diagnostics["is_weight"].requires_grad, "IS weight must be detached"
    # The stabilized loss must not create a gradient path through log_prob.
    assert loss.grad_fn is None


def test_topk_is_weight_requires_log_probs():
    per_token = torch.tensor([[1.0]])
    try:
        stabilize_topk_sdl_per_token_loss(per_token, use_is_weight=True)
    except ValueError as exc:
        assert "use_is_weight" in str(exc)
    else:
        raise AssertionError("expected ValueError when log probs are missing")


def test_topk_clamp_then_is_weight_compose():
    per_token = torch.tensor([[30.0, 2.0]])
    log_prob = torch.tensor([[-1.0, 0.5]])
    old_log_prob = torch.tensor([[0.0, 0.0]])

    loss, diagnostics = stabilize_topk_sdl_per_token_loss(
        per_token,
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        use_is_weight=True,
        loss_clamp=20.0,
    )

    expected = torch.tensor([[20.0 * torch.exp(torch.tensor(-1.0)).item(), 2.0 * torch.exp(torch.tensor(0.5)).item()]])
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(diagnostics["clamped_mask"], torch.tensor([[1.0, 0.0]]))


def test_dynamic_micro_batch_loss_metric_matches_backward_weighting():
    metric_sums = {}
    mini_batch_size = 8
    for micro_batch_size, loss in ((5, 2.0), (3, 10.0)):
        scale = _micro_batch_loss_scale_factor(
            use_dynamic_bsz=True,
            micro_batch_size=micro_batch_size,
            mini_batch_size=mini_batch_size,
            gradient_accumulation=None,
        )
        _accumulate_scaled_loss_metric(metric_sums, "actor/sdl_loss", loss, scale)

    assert metric_sums["actor/sdl_loss"] == 5.0


def test_fixed_micro_batch_loss_metric_matches_gradient_accumulation():
    metric_sums = {}
    for loss in (1.0, 3.0, 5.0, 7.0):
        scale = _micro_batch_loss_scale_factor(
            use_dynamic_bsz=False,
            micro_batch_size=2,
            mini_batch_size=8,
            gradient_accumulation=4,
        )
        _accumulate_scaled_loss_metric(metric_sums, "actor/sdl_loss", loss, scale)

    assert metric_sums["actor/sdl_loss"] == 4.0


def test_sdl_ratio_metrics_are_aggregated_by_counts_not_micro_batch_mean():
    metric_parts = {}
    _accumulate_ratio_metric(metric_parts, "actor/sdl_weighted_token_ratio", 1.0, 2.0)
    _accumulate_ratio_metric(metric_parts, "actor/sdl_weighted_token_ratio", 9.0, 10.0)

    metrics = _finalize_ratio_metrics(metric_parts)

    assert metrics["actor/sdl_weighted_token_ratio"] == 10.0 / 12.0


def test_actor_topk_producer_and_consumer_share_response_alignment_without_rmpad(monkeypatch):
    def cpu_logprobs(logits, labels, inplace_backward=True):
        del inplace_backward
        return F.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    monkeypatch.setattr("verl.workers.actor.dp_actor.logprobs_from_logits", cpu_logprobs)
    config = OmegaConf.create(
        {
            "use_remove_padding": False,
            "use_fused_kernels": False,
            "ulysses_sequence_parallel_size": 1,
            "use_torch_compile": False,
            "sdl_topk_chunk_size": 2,
        }
    )
    actor = DataParallelPPOActor(config, _TokenTableModel(vocab_size=7))
    actor.device_name = "cpu"
    micro_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3, 4], [1, 0, 3, 2, 5]]),
        "responses": torch.tensor([[3, 4], [2, 5]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
        "position_ids": torch.arange(5).repeat(2, 1),
    }

    _, _, teacher_outputs = actor._forward_micro_batch(micro_batch, temperature=1.0, return_topk=3)
    _, _, student_outputs = actor._forward_micro_batch(
        micro_batch,
        temperature=1.0,
        teacher_topk_ids=teacher_outputs["teacher_topk_ids"],
        teacher_topk_log_probs=teacher_outputs["teacher_topk_log_probs"],
    )

    assert teacher_outputs["teacher_topk_ids"].shape == (2, 2, 3)
    assert teacher_outputs["teacher_topk_ids"].dtype == torch.int32
    assert teacher_outputs["teacher_topk_log_probs"].dtype == torch.float32
    torch.testing.assert_close(
        student_outputs["topk_sdl_per_token"],
        torch.zeros(2, 2),
        atol=1e-6,
        rtol=0,
    )
