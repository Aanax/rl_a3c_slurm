# Split-encoder Shared-Diff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two split-encoder oracle model variants where the actor/oracle branch consumes the temporal diff of `shared_encoder` features (current − previous), while the critic branch keeps the current shared features.

**Architecture:** Two new subclasses of the existing split-encoder classes store the previous `shared_encoder` output as per-worker model state (`prev_shared`, detached) and feed `shared − prev_shared` into `actor_oracle_encoder`. State is reset at episode boundaries via `player_util`. Input becomes single-channel via a config-level wrapper swap (`NormalizedEnv_orig` + `normalization_alpha=0.999`), which makes the decoder and restoration target 1-channel automatically. No changes to `environment.py`, `train.py`, or `eval.py` (isinstance checks already match subclasses).

**Tech Stack:** PyTorch (A3C, Hogwild multiprocessing), runs on the kiae server via Slurm — NOT locally.

**Execution note:** This project runs on the server. Do not run code/tests locally. The optional smoke check in Task 5 is a snippet to run on the server.

**Reference spec:** `docs/superpowers/specs/2026-06-07-split-shared-diff-design.md`

---

### Task 1: Add `A3CRules2378OracleSplitEncodersSharedDiff` to `model.py`

**Files:**
- Modify: `src/model.py` (append at end of file, after `A3CRules2378OracleSplitEncodersIntrinsicCritic`)

- [ ] **Step 1: Append the new class**

Add to the end of `src/model.py`:

```python


class A3CRules2378OracleSplitEncodersSharedDiff(A3CRules2378OracleSplitEncoders):
    """Split-encoder oracle model where the actor/oracle branch consumes the temporal
    difference of shared-encoder features (current - previous), while the critic branch
    consumes the current shared features.

    `prev_shared` is per-worker model state (each train worker holds a local model copy).
    It is stored detached (no BPTT across steps) and reset to None at episode boundaries
    (see player_util), so the first frame of an episode yields a zero diff
    ("previous == current").
    """
    def __init__(self, num_inputs, action_space, args):
        super(A3CRules2378OracleSplitEncodersSharedDiff, self).__init__(num_inputs, action_space, args)
        self.prev_shared = None

    def forward(self, inputs, hx, cx, mem=None):
        shared, _, _, _ = self.shared_encoder(inputs)
        shared = shared.view(shared.size(0), 64, 4, 4)

        if self.prev_shared is None:
            diff = torch.zeros_like(shared)
        else:
            diff = shared - self.prev_shared
        self.prev_shared = shared.detach()

        actor_oracle_feat = F.relu(self.actor_oracle_encoder(diff))
        critic_feat = F.relu(self.critic_encoder(shared))

        actor_oracle_flat = actor_oracle_feat.view(actor_oracle_feat.size(0), -1)
        critic_flat = critic_feat.view(critic_feat.size(0), -1)

        if self.monitor_s:
            self.s_values.append(shared.detach().cpu())

        x_restored = self.decoder(actor_oracle_feat)

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])
        return self.critic_linear(critic_flat), self.actor_linear(actor_oracle_flat), hx, cx, x_restored
```

- [ ] **Step 2: Verify the diff routing differs from the parent**

Read back the appended class and confirm:
- `actor_oracle_encoder` receives `diff` (NOT `shared`).
- `critic_encoder` receives `shared`.
- `self.prev_shared` is assigned `shared.detach()` every forward.
- Layer signatures are unchanged (`diff` has the same shape `[B,64,4,4]` as `shared`).

- [ ] **Step 3: Commit**

```bash
git add src/model.py
git commit -m "feat(model): add A3CRules2378OracleSplitEncodersSharedDiff (shared-feature temporal diff into actor/oracle)"
```

---

### Task 2: Add `A3CRules2378OracleSplitEncodersSharedDiffIntrinsicCritic` to `model.py`

**Files:**
- Modify: `src/model.py` (append after the class from Task 1)

- [ ] **Step 1: Append the intrinsic-critic variant**

Add to the end of `src/model.py`:

```python


class A3CRules2378OracleSplitEncodersSharedDiffIntrinsicCritic(A3CRules2378OracleSplitEncodersIntrinsicCritic):
    """SharedDiff variant with an additional intrinsic critic head.

    Inherits `critic_linear_intrinsic` from A3CRules2378OracleSplitEncodersIntrinsicCritic
    and overrides forward with the shared-feature temporal-diff routing (same as
    A3CRules2378OracleSplitEncodersSharedDiff). The intrinsic value is read from the
    actor/oracle branch.
    """
    def __init__(self, num_inputs, action_space, args):
        super(A3CRules2378OracleSplitEncodersSharedDiffIntrinsicCritic, self).__init__(num_inputs, action_space, args)
        self.prev_shared = None

    def forward(self, inputs, hx, cx, mem=None):
        shared, _, _, _ = self.shared_encoder(inputs)
        shared = shared.view(shared.size(0), 64, 4, 4)

        if self.prev_shared is None:
            diff = torch.zeros_like(shared)
        else:
            diff = shared - self.prev_shared
        self.prev_shared = shared.detach()

        actor_oracle_feat = F.relu(self.actor_oracle_encoder(diff))
        critic_feat = F.relu(self.critic_encoder(shared))

        actor_oracle_flat = actor_oracle_feat.view(actor_oracle_feat.size(0), -1)
        critic_flat = critic_feat.view(critic_feat.size(0), -1)

        if self.monitor_s:
            self.s_values.append(shared.detach().cpu())

        x_restored = self.decoder(actor_oracle_feat)

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])
        value_intrinsic = self.critic_linear_intrinsic(actor_oracle_flat)
        return (
            self.critic_linear(critic_flat),
            self.actor_linear(actor_oracle_flat),
            hx,
            cx,
            x_restored,
            value_intrinsic,
        )
```

Note: forward duplication between the two new classes mirrors the existing pattern in
`A3CRules2378OracleSplitEncoders` / `...IntrinsicCritic` (the base pair also duplicates the
full forward). Keep it consistent rather than introducing a new shared helper.

- [ ] **Step 2: Verify inheritance gives free isinstance coverage**

Confirm the class extends `A3CRules2378OracleSplitEncodersIntrinsicCritic`. Because of this:
- `src/player_util.py` line ~57 isinstance tuple (contains `A3CRules2378OracleSplitEncodersIntrinsicCritic`) matches this subclass → 6-output unpacking works.
- `src/train.py` line ~207 isinstance tuple matches this subclass → `R_intrinsic` / `w_intrinsic=1.0` path used.
- `src/eval.py` line ~173 oracle-models tuple (contains the split base classes) matches both new subclasses.

No edits to `train.py` / `eval.py` are required. Do not add them.

- [ ] **Step 3: Commit**

```bash
git add src/model.py
git commit -m "feat(model): add A3CRules2378OracleSplitEncodersSharedDiffIntrinsicCritic"
```

---

### Task 3: Reset `prev_shared` at episode boundaries in `player_util.py`

**Files:**
- Modify: `src/player_util.py` (`action_train` done-block ~lines 98-102, `action_test` done-block ~lines 129-135)

- [ ] **Step 1: Add reset in `action_train`**

Find this block in `action_train`:

```python
        if self.done:
            if hasattr(self.model, 'running_mem'):
                self.model.running_mem = torch.zeros_like(self.model.running_mem)
            if hasattr(self.model, 'prev_x_conv'):
                self.model.prev_x_conv = None
```

Replace with:

```python
        if self.done:
            if hasattr(self.model, 'running_mem'):
                self.model.running_mem = torch.zeros_like(self.model.running_mem)
            if hasattr(self.model, 'prev_x_conv'):
                self.model.prev_x_conv = None
            if hasattr(self.model, 'prev_shared'):
                self.model.prev_shared = None
```

- [ ] **Step 2: Add reset in `action_test`**

Find this block in `action_test` (inside the `try`):

```python
                try:
                    if hasattr(self.model, 'running_mem'):
                        self.model.running_mem = torch.zeros_like(self.model.running_mem)
                    if hasattr(self.model, 'prev_x_conv'):
                        self.model.prev_x_conv = None
                except:
                    pass
```

Replace with:

```python
                try:
                    if hasattr(self.model, 'running_mem'):
                        self.model.running_mem = torch.zeros_like(self.model.running_mem)
                    if hasattr(self.model, 'prev_x_conv'):
                        self.model.prev_x_conv = None
                    if hasattr(self.model, 'prev_shared'):
                        self.model.prev_shared = None
                except:
                    pass
```

- [ ] **Step 3: Commit**

```bash
git add src/player_util.py
git commit -m "feat(player_util): reset prev_shared at episode boundaries (train + test)"
```

---

### Task 4: Add experiment configs

**Files:**
- Create: `configs/run_config_oracle_split_shared_diff_8_64_v2.ini`
- Create: `configs/run_config_oracle_split_shared_diff_intrinsic_8_64_v2.ini`

- [ ] **Step 1: Create the SharedDiff config**

Create `configs/run_config_oracle_split_shared_diff_8_64_v2.ini`:

```ini
[DEFAULT]
experiment_name=1024fc_A3CRules2378OracleSplitEncodersSharedDiff_32w_8_64_v2
lr=0.0001
entropy_coef=0.00
value_coef=0.5
gamma=0.984375
gamma_memory=0.875
gamma_restoration=0.9
relu_g_const=True
tau=1.00
seed=0
workers=32
num_steps=32
max_episode_length=10000
env=PongNoFrameskip-v4
shared_optimizer=True
load=False
save_max=False
optimizer=Adam
load_model_dir=trained_models/
save_model_dir=trained_models/
log_dir=logs/
gpu_ids=-1
amsgrad=False
skip_rate=4
hidden_size=1024
tensorboard_logger=False
gif_image_save_frequency=100000
env_config=configs/envs_config.json
distributed_step_size=
input_normalization_class=NormalizedEnv_orig
normalization_alpha=0.999
model_type=A3CRules2378OracleSplitEncodersSharedDiff
monitor_s=True
monitor_s_save_interval=40
w_kld_loss=0.0
w_restoration_loss=1.0
monitor_losses=True
monitor_cosine_const=True
total_steps_stop=15000000
```

- [ ] **Step 2: Create the intrinsic-critic config**

Create `configs/run_config_oracle_split_shared_diff_intrinsic_8_64_v2.ini` — identical to Step 1 except `experiment_name` and `model_type`:

```ini
[DEFAULT]
experiment_name=1024fc_A3CRules2378OracleSplitEncodersSharedDiffIntrinsicCritic_32w_8_64_v2
lr=0.0001
entropy_coef=0.00
value_coef=0.5
gamma=0.984375
gamma_memory=0.875
gamma_restoration=0.9
relu_g_const=True
tau=1.00
seed=0
workers=32
num_steps=32
max_episode_length=10000
env=PongNoFrameskip-v4
shared_optimizer=True
load=False
save_max=False
optimizer=Adam
load_model_dir=trained_models/
save_model_dir=trained_models/
log_dir=logs/
gpu_ids=-1
amsgrad=False
skip_rate=4
hidden_size=1024
tensorboard_logger=False
gif_image_save_frequency=100000
env_config=configs/envs_config.json
distributed_step_size=
input_normalization_class=NormalizedEnv_orig
normalization_alpha=0.999
model_type=A3CRules2378OracleSplitEncodersSharedDiffIntrinsicCritic
monitor_s=True
monitor_s_save_interval=40
w_kld_loss=0.0
w_restoration_loss=1.0
monitor_losses=True
monitor_cosine_const=True
total_steps_stop=15000000
```

- [ ] **Step 3: Commit**

```bash
git add configs/run_config_oracle_split_shared_diff_8_64_v2.ini configs/run_config_oracle_split_shared_diff_intrinsic_8_64_v2.ini
git commit -m "feat(configs): add shared-diff split experiment configs (1-channel NormalizedEnv_orig, alpha=0.999)"
```

---

### Task 5: Add Slurm launch scripts + server smoke check

**Files:**
- Create: `slurm_scripts/our/parallel_oracle_split_shared_diff_8_64_v2.sh`
- Create: `slurm_scripts/our/parallel_oracle_split_shared_diff_intrinsic_8_64_v2.sh`

- [ ] **Step 1: Create the SharedDiff Slurm script**

Create `slurm_scripts/our/parallel_oracle_split_shared_diff_8_64_v2.sh`:

```bash
#!/bin/bash
set -euo pipefail

name_part="oracle_split_shared_diff_8_64_v2"

/s/ls4/users/dartl0l/goarl/rl_a3c_slurm/slurm_scripts/our/parallel_oracle_runner.sh "$@" \
  "${name_part}" \
  /s/ls4/users/dartl0l/goarl/rl_a3c_slurm/configs/run_config_oracle_split_shared_diff_8_64_v2.ini \
  > "run_ids_${name_part}.txt"
```

- [ ] **Step 2: Create the intrinsic-critic Slurm script**

Create `slurm_scripts/our/parallel_oracle_split_shared_diff_intrinsic_8_64_v2.sh`:

```bash
#!/bin/bash
set -euo pipefail

name_part="oracle_split_shared_diff_intrinsic_8_64_v2"

/s/ls4/users/dartl0l/goarl/rl_a3c_slurm/slurm_scripts/our/parallel_oracle_runner.sh "$@" \
  "${name_part}" \
  /s/ls4/users/dartl0l/goarl/rl_a3c_slurm/configs/run_config_oracle_split_shared_diff_intrinsic_8_64_v2.ini \
  > "run_ids_${name_part}.txt"
```

- [ ] **Step 3: Make scripts executable + commit**

```bash
chmod +x slurm_scripts/our/parallel_oracle_split_shared_diff_8_64_v2.sh slurm_scripts/our/parallel_oracle_split_shared_diff_intrinsic_8_64_v2.sh
git add slurm_scripts/our/parallel_oracle_split_shared_diff_8_64_v2.sh slurm_scripts/our/parallel_oracle_split_shared_diff_intrinsic_8_64_v2.sh
git commit -m "feat(slurm): add shared-diff split launch scripts"
```

- [ ] **Step 4: (Optional) Server-side smoke check**

Run ON THE SERVER (e.g. in a Jupyter cell or a quick python session in the repo `src/` dir),
NOT locally. Verifies forward shapes and the diff/reset behavior without launching training:

```python
import argparse, torch
import model as M

class _AS:  # fake action space
    n = 6

args = argparse.Namespace(hidden_size=1024, monitor_s=False, use_rmsnorm=False)
net = M.A3CRules2378OracleSplitEncodersSharedDiffIntrinsicCritic(1, _AS(), args)

x1 = torch.randn(1, 1, 80, 80)
out = net(x1, None, None)
assert len(out) == 6, len(out)
value, logit, hx, cx, x_restored, v_int = out
print("logit", logit.shape, "value", value.shape, "x_restored", x_restored.shape, "v_int", v_int.shape)

# First forward: prev_shared was None -> diff == 0 -> actor/oracle features come from zero diff.
# After a forward, prev_shared is populated and detached:
assert net.prev_shared is not None and not net.prev_shared.requires_grad

# Reset semantics (what player_util does on episode end):
net.prev_shared = None
out2 = net(x1, None, None)
assert len(out2) == 6
print("smoke OK")
```

Expected: prints shapes (`logit` = `[1,6]`, `value` = `[1,1]`, `x_restored` = `[1,1,80,80]`, `v_int` = `[1,1]`) and `smoke OK`, no assertion errors.

---

## Notes for the implementer

- **Do not** edit `src/environment.py`, `src/train.py`, or `src/eval.py`. The new models reuse `NormalizedEnv_orig` and are covered by existing isinstance checks via inheritance.
- **Do not** add local `python`/`pytest` execution steps — this project runs on the server only.
- The two `.ini` files differ only in `experiment_name` and `model_type`; the two Slurm scripts differ only in `name_part` and the config path.
