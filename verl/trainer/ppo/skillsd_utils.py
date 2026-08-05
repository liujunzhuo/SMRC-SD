"""
SkillSD (Skill-based Self-Distillation) utilities.

Provides the SDL (Self Distillation Loss) computation for the SkillSD algorithm.
Skill retrieval and teacher batch construction are reused from rlsd_utils.py
and rlsd_ray_trainer.py without modification.
"""

from typing import Optional

import torch

from verl.trainer.ppo.core_algos import agg_loss


def compute_topk_forward_kl_with_tail(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    include_tail: bool = True,
    eps: float = 1e-8,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute teacher-support Top-k forward KL with an optional tail bucket.

    The teacher's K explicit tokens and the aggregate probability outside that
    support form a K+1 categorical distribution.  This preserves normalization
    without materializing a full-vocabulary log-softmax and avoids the negative
    values produced by naively truncating a KL divergence.

    Args:
        student_logits: ``(..., vocab_size)`` current-policy logits. Gradients
            are retained only through this tensor.
        teacher_topk_ids: ``(..., K)`` teacher-support token ids.
        teacher_topk_log_probs: ``(..., K)`` detached teacher log-probabilities
            over the full vocabulary, not renormalized within Top-k.
        include_tail: Include one bucket for all vocabulary items outside the
            teacher support. Disabling it is intended only for a controlled
            truncated-loss ablation.
        eps: Numerical floor for logarithms of tail mass.
        chunk_size: Number of token rows processed together. Chunking avoids a
            full fp32 ``[..., vocab_size]`` temporary for large vocabularies.

    Returns:
        A per-token loss with shape ``student_logits.shape[:-1]`` and detached
        teacher/student support-mass diagnostics of the same shape.
    """
    if student_logits.shape[:-1] != teacher_topk_ids.shape[:-1]:
        raise ValueError(
            "student_logits and teacher_topk_ids prefix shapes must match, "
            f"got {tuple(student_logits.shape)} and {tuple(teacher_topk_ids.shape)}"
        )
    if teacher_topk_ids.shape != teacher_topk_log_probs.shape:
        raise ValueError(
            "teacher_topk_ids and teacher_topk_log_probs must have identical shapes, "
            f"got {tuple(teacher_topk_ids.shape)} and {tuple(teacher_topk_log_probs.shape)}"
        )
    if teacher_topk_ids.shape[-1] <= 0:
        raise ValueError("Top-k support must contain at least one token")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    original_shape = student_logits.shape[:-1]
    vocab_size = student_logits.shape[-1]
    flat_student_logits = student_logits.reshape(-1, vocab_size)
    flat_teacher_ids = teacher_topk_ids.reshape(-1, teacher_topk_ids.shape[-1])
    flat_teacher_log_probs = teacher_topk_log_probs.reshape(-1, teacher_topk_log_probs.shape[-1])
    loss_chunks = []
    diagnostic_chunks = {name: [] for name in ("teacher_mass", "student_mass", "teacher_tail_mass", "student_tail_mass")}

    for start in range(0, flat_student_logits.shape[0], chunk_size):
        end = min(start + chunk_size, flat_student_logits.shape[0])
        # Keep the reduction in fp32. In particular, bf16 rounding can make a
        # high-mass Top-k support sum slightly above one and destabilize tail KL.
        student_logits_fp32 = flat_student_logits[start:end].float()
        teacher_log_probs = flat_teacher_log_probs[start:end].detach().float()
        teacher_ids = flat_teacher_ids[start:end].long()
        student_log_z = torch.logsumexp(student_logits_fp32, dim=-1, keepdim=True)
        student_log_probs = torch.gather(student_logits_fp32, dim=-1, index=teacher_ids) - student_log_z

        teacher_probs = teacher_log_probs.exp()
        student_probs = student_log_probs.exp()
        teacher_mass = teacher_probs.sum(dim=-1)
        student_mass = student_probs.sum(dim=-1)

        loss = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        teacher_tail = (1.0 - teacher_mass).clamp(min=0.0, max=1.0)
        student_tail = (1.0 - student_mass).clamp(min=0.0, max=1.0)
        if include_tail:
            # p * log(p / q), with the continuous convention 0 * log(0 / q) = 0.
            loss = loss + teacher_tail * (
                torch.log(teacher_tail.clamp_min(eps)) - torch.log(student_tail.clamp_min(eps))
            )

        loss_chunks.append(loss)
        diagnostic_chunks["teacher_mass"].append(teacher_mass.detach())
        diagnostic_chunks["student_mass"].append(student_mass.detach())
        diagnostic_chunks["teacher_tail_mass"].append(teacher_tail.detach())
        diagnostic_chunks["student_tail_mass"].append(student_tail.detach())

    loss = torch.cat(loss_chunks, dim=0).reshape(original_shape)
    diagnostics = {
        name: torch.cat(chunks, dim=0).reshape(original_shape) for name, chunks in diagnostic_chunks.items()
    }
    return loss, diagnostics


def stabilize_topk_sdl_per_token_loss(
    per_token_loss: torch.Tensor,
    *,
    log_prob: Optional[torch.Tensor] = None,
    old_log_prob: Optional[torch.Tensor] = None,
    use_is_weight: bool = False,
    loss_clamp: float = 0.0,
    is_log_ratio_clamp: float = 10.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply optional stabilizers to the directly backpropagated Top-k SDL loss.

    Two independent, opt-in measures:

    - ``loss_clamp`` (> 0 enables): per-token upper clamp of the forward-KL
      loss. When the student assigns near-zero mass to the whole teacher
      support, the K+1 forward KL and its gradient blow up; clamping caps the
      per-token contribution while the returned ``clamped_mask`` diagnostic
      exposes how often this fires (a persistent high ratio signals a
      teacher/student distribution break, which should be inspected rather
      than silently averaged away).
    - ``use_is_weight``: multiply by a **detached** truncated importance
      weight ``min(exp(log_prob - old_log_prob), exp(is_log_ratio_clamp))``.
      Unlike the chosen-token K3 estimator (where the non-detached ``rho * k3``
      product forms the correct score-function estimator), the Top-k forward
      KL is already directly differentiable; a detached weight only reweights
      off-policy tokens during minibatch updates without adding a
      score-function term.

    Returns the stabilized per-token loss and detached diagnostics
    (``clamped_mask`` and/or ``is_weight`` when the respective measure is on).
    """
    diagnostics: dict[str, torch.Tensor] = {}
    loss = per_token_loss
    if loss_clamp and loss_clamp > 0.0:
        diagnostics["clamped_mask"] = (loss.detach() > loss_clamp).to(loss.dtype)
        loss = loss.clamp(max=float(loss_clamp))
    if use_is_weight:
        if log_prob is None or old_log_prob is None:
            raise ValueError("use_is_weight=True requires log_prob and old_log_prob")
        log_ratio = (log_prob - old_log_prob).clamp(max=float(is_log_ratio_clamp))
        is_weight = torch.exp(log_ratio).detach()
        diagnostics["is_weight"] = is_weight
        loss = loss * is_weight
    return loss, diagnostics


def aggregate_sdl_per_token_loss(
    per_token_loss: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    normalization_mask: Optional[torch.Tensor] = None,
    loss_normalization: str = "selected_token_mean",
) -> torch.Tensor:
    """Aggregate any directly backpropagated SDL token loss consistently."""
    loss_normalization = str(loss_normalization or "selected_token_mean").lower()
    if loss_normalization in ("selected", "selected_token", "selected_token_mean"):
        return agg_loss(loss_mat=per_token_loss, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    if loss_normalization in ("response", "response_token", "response_token_mean", "original_response_token_mean"):
        if normalization_mask is None:
            raise ValueError("loss_normalization=response_token_mean requires normalization_mask")
        denom = normalization_mask.float().sum().clamp(min=1.0)
        return torch.sum(per_token_loss * response_mask) / denom
    raise ValueError(
        f"Unsupported SDL loss_normalization={loss_normalization!r}; "
        "expected selected_token_mean or response_token_mean"
    )


def compute_candidate_ce_loss(
    candidate_log_probs: torch.Tensor,
    target_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    normalization_token_count: torch.Tensor,
    loss_normalization: str = "response_token_mean",
) -> torch.Tensor:
    """Compute CE on matcher-provided action-only completions.

    ``response_token_mean`` uses the sampled on-policy response-token count as
    the denominator. This keeps the auxiliary coefficient and update mass
    directly comparable to sparse Path-OPD SDL runs.
    """
    if candidate_log_probs.shape != target_mask.shape:
        raise ValueError(
            "candidate_log_probs and target_mask must have identical shapes, "
            f"got {tuple(candidate_log_probs.shape)} and {tuple(target_mask.shape)}"
        )
    if sample_weight.ndim != 1 or sample_weight.shape[0] != candidate_log_probs.shape[0]:
        raise ValueError(
            "sample_weight must have shape [batch], "
            f"got {tuple(sample_weight.shape)} for batch {candidate_log_probs.shape[0]}"
        )
    if normalization_token_count.ndim != 1 or normalization_token_count.shape[0] != candidate_log_probs.shape[0]:
        raise ValueError(
            "normalization_token_count must have shape [batch], "
            f"got {tuple(normalization_token_count.shape)} for batch {candidate_log_probs.shape[0]}"
        )

    weighted_mask = target_mask.to(candidate_log_probs.dtype) * sample_weight.to(
        dtype=candidate_log_probs.dtype,
        device=candidate_log_probs.device,
    ).clamp(min=0.0).unsqueeze(-1)
    numerator = -(candidate_log_probs * weighted_mask).sum()
    loss_normalization = str(loss_normalization or "response_token_mean").lower()
    if loss_normalization in ("selected", "selected_token", "selected_token_mean"):
        denominator = weighted_mask.sum().clamp(min=1.0)
    elif loss_normalization in (
        "response",
        "response_token",
        "response_token_mean",
        "original_response_token_mean",
    ):
        denominator = normalization_token_count.to(
            dtype=candidate_log_probs.dtype,
            device=candidate_log_probs.device,
        ).sum().clamp(min=1.0)
    else:
        raise ValueError(
            f"Unsupported candidate CE normalization={loss_normalization!r}; "
            "expected selected_token_mean or response_token_mean"
        )
    return numerator / denominator


def compute_sdl_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    normalization_mask: Optional[torch.Tensor] = None,
    loss_normalization: str = "selected_token_mean",
) -> torch.Tensor:
    """
    Compute the Self Distillation Loss (SDL) for SkillSD.

    Args:
        student_log_probs: (bs, response_length) — log π_θ(y_t | x, y_<t).
            Current policy forward pass; retains gradients.
        teacher_log_probs: (bs, response_length) — log π_θ(y_t | x ⊕ S(x), y_<t).
            Frozen (no grad). Teacher sees skill-augmented input.
        old_log_probs: (bs, response_length) — log π_old(y_t | x, y_<t).
            Frozen (no grad). Rollout-time policy.
        response_mask: (bs, response_length) — mask for tokens selected for SDL.
        loss_agg_mode: aggregation mode passed to agg_loss.
        normalization_mask: optional denominator mask. Used by response_token_mean
            to keep sparse SDL masks from increasing per-token update strength.
        loss_normalization: selected_token_mean keeps the legacy behavior and
            averages over response_mask. response_token_mean sums the selected
            SDL loss but divides by normalization_mask, usually the original
            response token mask.

    Returns:
        sdl_loss: scalar — the aggregated SDL loss.
    """
    teacher_log_probs = teacher_log_probs.detach()
    old_log_probs = old_log_probs.detach()

    ell = student_log_probs - teacher_log_probs

    neg_ell_clamped = (-ell).clamp(max=20.0)
    k3 = torch.exp(neg_ell_clamped) - 1.0 + ell

    log_rho_on = (student_log_probs - old_log_probs).clamp(max=10.0)
    rho_on = torch.exp(log_rho_on)

    sdl_per_token = rho_on * k3

    return aggregate_sdl_per_token_loss(
        per_token_loss=sdl_per_token,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        normalization_mask=normalization_mask,
        loss_normalization=loss_normalization,
    )
