# Concat Memdiff Raw Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add unnormalized memory-difference routing and launch a no-split, no-oracle experiment with it.

**Architecture:** Reuse `_SharedFeatureDiffMixin` and its existing differentiable memory accumulator. Add one explicit mode whose only difference is the unnormalized `(1-gamma_memory)` weight; preserve `concat_memdiff`.

**Tech Stack:** Python, PyTorch, INI configuration, Slurm.

## Global Constraints

- Existing `concat_memdiff` behavior must not change.
- New mode is `concat_memdiff_raw`.
- Runtime verification happens only on kiae.
- Do not create git commits.

---

### Task 1: Add Raw Memory-Difference Routing

**Files:**
- Modify: `src/model.py`

**Interfaces:**
- Consumes: `_SharedFeatureDiffMixin.memdiff_sum`, `gamma_memory`, `_branch_inputs(shared)`.
- Produces: `actor_input_mode=concat_memdiff_raw` and `critic_input_mode=concat_memdiff_raw`.

- [ ] Add `concat_memdiff_raw` to `_VALID_MODES`.
- [ ] Compute raw memory difference as:

```python
raw_memdiff = (
    torch.zeros_like(shared)
    if self.memdiff_sum is None or self.memdiff_count == 0
    else shared - self.memdiff_sum * (1.0 - self.gamma_memory)
)
```

- [ ] Update the memory accumulator once per forward, shared by normalized and raw modes.
- [ ] Route `concat([shared, raw_memdiff])` for the new mode.

### Task 2: Add Experiment Files

**Files:**
- Create: `configs/run_config_oracle_nosplit_concat_memdiff_raw_both_no_oracle_8_64_v2.ini`
- Create: `slurm_scripts/our/parallel_oracle_nosplit_concat_memdiff_raw_both_no_oracle_8_64_v2.sh`

**Configuration:**

```ini
model_type=A3CRules2378OracleNoSplitSharedDiff
actor_input_mode=concat_memdiff_raw
critic_input_mode=concat_memdiff_raw
gamma_memory=0.875
w_restoration_loss=0.0
```

### Task 3: Launch and Verify

- [ ] Rsync `src`, `configs`, and `slurm_scripts` to kiae.
- [ ] Launch 10 runs with the new parallel launcher.
- [ ] Report job IDs and Slurm states.
- [ ] Once a run starts, verify zero Python Tracebacks and advancing loss rows without reading complete logs.
