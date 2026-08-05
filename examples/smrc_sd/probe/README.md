# Fixed-anchor teacher intervention probe

Offline mechanism probe used in the paper (1,200 ALFWorld turns: 400 each in strata A / B / C).

It does **not** start Ray training. It builds a fixed anchor manifest from GRPO rollout dumps, then scores teacher contexts with FSDP `torchrun`.

## Strata

- **A**: supported turns where the sampled action agrees with the matched candidate
- **B**: supported turns where the sampled action diverges from the candidate (balanced success/failure trajectories)
- **C**: unsupported turns with a valid progress-making sampled action from a successful trajectory

## Run

```bash
# from repo root, after GRPO training artifacts exist
export GRPO_ROLLOUT_DIR=logs/smrc_sd_alfworld/rollouts/grpo_alfworld_qwen3_1p7b
export GRPO250_ACTOR=checkpoints/smrc_sd_alfworld/grpo_alfworld_qwen3_1p7b/global_step_250/actor
bash examples/smrc_sd/probe/run_anchor_probe.sh
```

Useful overrides: `PROBE_ROOT`, `MANIFEST`, `BASE_MODEL`, `BATCH_SIZE`, `NPROC`, `CUDA_VISIBLE_DEVICES`.

## Note on internal keys

The manifest uses the `smrc` context key and `SMRC-*` anchor IDs, matching the paper name **SMRC-SD**.
