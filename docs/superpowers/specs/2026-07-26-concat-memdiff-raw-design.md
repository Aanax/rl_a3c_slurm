# Concat Memdiff Raw

## Goal

Add an unnormalized memory-difference routing mode without changing the existing
`concat_memdiff` behavior.

## Mode

New `actor_input_mode` / `critic_input_mode` value:

```ini
concat_memdiff_raw
```

At step `t`, before updating the memory:

```text
memory = s[t-1] + gamma_memory*s[t-2] + ...
memdiff_raw = s[t] - (1 - gamma_memory)*memory
branch_input = concat([s[t], memdiff_raw])
```

For the first state, `memdiff_raw` is zero. The memory remains differentiable
inside a rollout and is detached at the batch boundary by the existing
`train.py` logic. It is reset at the episode boundary.

The existing normalized `concat_memdiff` remains:

```text
s[t] - memory*(1-gamma_memory)/(1-gamma_memory**t)
```

## Experiment

Create and launch 10 runs with:

```ini
model_type=A3CRules2378OracleNoSplitSharedDiff
actor_input_mode=concat_memdiff_raw
critic_input_mode=concat_memdiff_raw
gamma_memory=0.875
w_restoration_loss=0.0
```

Use a new experiment/config/launcher name containing
`nosplit_concat_memdiff_raw_both_no_oracle`.

## Verification

After submission, verify Slurm status and count Python tracebacks without
reading complete logs. Once running, confirm that training output or loss CSV
is advancing past several rollout boundaries.
