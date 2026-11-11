# P2-03A — Action-conditioned JEPA: Architecture Specification & Staged Plan

> **Design document only (Phase 0).** No code is implemented here. Every
> interface below is derived from the audited source; see
> `P2_03A_ARCHITECTURE_AUDIT.md` for `file:line` evidence.
> Implementation begins only after review of the audit + this plan.

## 1. Research objective

Move from unconditional latent prediction to intervention-conditioned latent
prediction on a **controlled physical system**, keeping SciNet++
domain-independent:

```
legacy / A2 infrastructure:   z_{t+1} = f_theta(z_t)
P2-03A:                        z_{t+1} = f_theta(z_t, u_t)
```

| | P2-02 (done) | P2-03A (this phase) |
|---|---|---|
| Space | raw state `x` | latent `z` |
| Input | `x_t, u_t` | `z_t, u_t` |
| Output | `x_hat_{t+1}` | `z_hat_{t+1}` |
| Model | `ResidualStatePredictor` | `Encoder + ConditionalPredictor + EMA target` |
| Question | does action add predictive value? | can a **representation** learn action-conditioned dynamics? |
| Domain | Pusher-v1 (arm kinematics) | torque-controlled pendulum (controlled physics) |

P2-02 already answered its question positively (B2 skill 0.8749). P2-03A must
**not** be justified by that latent loss alone; the Linear Probe and the G4
gate are what make the comparison meaningful.

## 2. Data contract and EXACT window indexing

### 2.1 Episode layout

For one episode of length `L`:

```
observations : x[0], x[1], ..., x[L-1]        x_t ∈ R^D   (D = 2: theta, omega)
conditions   : u[0], u[1], ..., u[L-2]        u_t ∈ R^U   (U = 1: torque)
```

Fixed A1 relation: **`(x_t, u_t) → x_{t+1}`**. `u_t` is the action applied at
state `x_t`; it is NOT the action that led to `x_t`.

Physical parameters `(m, L, b)` are per-episode constants recorded in the
manifest; the integrator is deterministic RK4 with fixed `dt`.

### 2.2 Adjacent shifted windows (the critical alignment contract)

Let `K = block_size`, and let a window start at `s` with `0 ≤ s ≤ L-K-1`.
Use A1's `dynamics_batch_from_windows(window_len=K+1, stride=1)`, which
returns, for each row, `obs[s : s+K+1]` and `cond[s : s+K]` and guarantees the
window never crosses an episode boundary (`contracts.py:241-312`).

Define `t = s + K - 1` (the LAST state of the context block). Then:

| Tensor | Definition | Shape | Meaning |
|---|---|---|---|
| `context_block` | `flatten(obs[s : s+K])` | `[B, K*D]` | `x_{t-K+1} … x_t` |
| `action_t` | `cond[s+K-1]` | `[B, U]` | **`u_t`**, applied at `x_t` |
| `target_block` | `flatten(obs[s+1 : s+K+1])` | `[B, K*D]` | `x_{t-K+2} … x_{t+1}` (shifted +1) |

Identities that MUST hold and MUST be unit-tested:

```
context_block[..., -D:]      == x_t
target_block[..., -D:]       == x_{t+1}
target_block[..., :-D]       == context_block[..., D:]      (shift by exactly 1)
action_t                     == u_t  where (x_t, u_t) -> x_{t+1}
every window's episode_id is single-valued (no cross-episode stitching)
```

For horizon `k` (multi-step):

```
target_k_block = flatten(obs[s+k : s+k+K+1])       # start shifted by k
action_seq     = cond[s+K-1 : s+K-1+k]             # u_t ... u_{t+k-1}
z_pred_k      = ConditionalPredictor.rollout(z_context, action_seq)[:, k-1]
z_target_k    = stop_gradient(target_encoder(target_k_block))
```

`ConditionalPredictor.rollout` (`conditional_predictor.py:148-197`) already feeds
predicted latents back and preserves the graph, so **no ground-truth latent is
re-injected**. Ground-truth latents appear only as EMA-target labels.

### 2.3 Why a new module is still needed

`dynamics_batch_from_windows` returns a `DynamicsBatch` of `(obs-window,
cond-window)` pairs, not the triple `(context_block, action_t, target_block)`,
and does not flatten to `[B, K*D]`. `trajectory_windows.py` is therefore a
**thin adapter** over it — no duplicate window/placement logic, no new
cross-episode risk. `BlockSequenceDataset` (`data/dataset.py:286`) is NOT used
because its blocks are non-overlapping and action-free.

## 3. Model: `ActionConditionedWorldModel` (new, independent)

```
models/action_conditioned_world_model.py   (NEW — never edit WorldModel)
```

Constructor (proposed, all configurable):

```python
class ActionConditionedWorldModel(nn.Module):
    def __init__(self, block_size, state_dim, condition_dim,
                 latent_dim=8, encoder_hidden=(64, 32),
                 predictor_hidden=(64, 64), encoder_type="mlp",
                 predictor=None, ema_momentum=0.996):
        self.encoder = Encoder(block_size, state_dim, encoder_hidden,
                              latent_dim, kind=encoder_type)       # REUSE
        self.predictor = predictor or ConditionalPredictor(         # REUSE
            latent_dim, condition_dim, predictor_hidden)
        self.target_encoder = copy.deepcopy(self.encoder)            # EMA PATTERN
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.is_conditioned = isinstance(predictor, ...)
```

Methods:

| Method | Behaviour |
|---|---|
| `encode(block_flat)` | online encoder, differentiable |
| `encode_target(block_flat)` | `@torch.no_grad()` EMA encoder (mirrors `world_model.py:76-79`) |
| `predict_step(z, condition)` | `ConditionalPredictor(z, condition)`; for the E1 ablation the injected legacy `Predictor(z)` is called with no condition |
| `rollout_latent(z0, conditions)` | delegates to `ConditionalPredictor.rollout` |
| `update_target_encoder(momentum)` | `@torch.no_grad()` EMA copy (`world_model.py:128-134`) |
| `forward(context, action, targets_by_horizon, horizons)` | returns `(preds_by_horizon, targets_by_horizon, z_context)` |

**Checkpoint isolation:** new checkpoints are separate files with a
`model_class = "ActionConditionedWorldModel"` tag; a two-way load test must
prove legacy checkpoints still load into `WorldModel` with `strict=True` and
that new checkpoints cannot be silently loaded into `WorldModel` (map §4).

**E1 unconditioned ablation:** instantiate legacy `Predictor(z)` and inject it.
Instantiation is not modification, so §8 is respected. `condition=None` must
NEVER be passed to A2's `ConditionalPredictor` (it rejects it by design,
`conditional_predictor.py:102-108`). E4 (zero action) is a DIFFERENT experiment:
same conditional architecture with a genuine all-zero tensor.

## 4. Training

### 4.1 Single-step first (P2)

```python
z_context = model.encode(context_block)                  # [B, L]
z_target  = model.encode_target(target_block).detach()   # [B, L], no grad
z_pred    = model.predictor(z_context, action_t)        # [B, L]

pred_l, _ = prediction_loss({1: z_pred}, {1: z_target}, [1])   # utils/losses.py:27
vic_l = vicreg_loss(z_context, var_weight, cov_weight)          # utils/losses.py:83
loss  = pred_weight * pred_l + vic_l

optimizer.zero_grad(); loss.backward(); optimizer.step()
model.update_target_encoder(ema_momentum(step, warmup_steps, base_momentum))
```

Only `requires_grad=True` parameters enter the optimizer (mirrors
`train.py:141-142`). VICReg acts on `z_context` only, as in the legacy loop
(`train.py:174-178`).

### 4.2 Multi-step only after single-step passes (P4)

Use `preds_by_horizon = {k: rollout[:, k-1] for k in horizons}` and
`targets_by_horizon = {k: encode_target(target_k_block).detach()}` with
`horizons = [1, 2, 5, 10]`, then `prediction_loss(preds, targets, horizons)`
with weights `w_k` from config. Keep VICReg unchanged. Do **not** introduce
Transformer encoders or extra regularisers in this phase.

### 4.3 Logged every epoch

EMA momentum, VICReg coefficients, per-dimension latent variance, covariance
off-diagonal magnitude, prediction loss per horizon, gradient norms, seed,
device, dataset version, config hash, git commit, duration. Exactly one unique
`experiment_id` per `(config_hash, variant, seed)`; never overwrite.

### 4.4 First-round config (design values, NOT validated results)

| Key | Value |
|---|---|
| `latent_dim` | 8 (or 16 — Q4) |
| `encoder_hidden` / `predictor_hidden` | `(64,32)` / `(64,64)` |
| `block_size` | 10 |
| `batch_size` | 128 |
| `lr` / `weight_decay` | `1e-3` / `1e-4` |
| `seeds` | 5 independent model seeds (dataset_seed strictly separate) |
| `device` | CPU smoke first; GPU optional |

## 5. Six experiments (E0–E5)

| ID | Encoder | Predictor | Action | Role |
|---|---|---|---|---|
| E0 | — | Persistence | — | lowest baseline (Q1: define precisely) |
| E1 | MLP | unconditioned | none | ordinary JEPA |
| E2 | MLP | conditional | correct | **target model** |
| E3 | MLP | conditional | shuffled | negative control |
| E4 | MLP | conditional | zero tensor | action-contribution check |
| E5 | MLP | conditional | correct | multi-step rollout |

All of E1–E5 share capacity, epochs, batch, seeds and the same train/validation
protocol. E3 uses a fixed-seed `shuffle_conditions` (generic helper from
`models/state_dynamics.py`, no domain constants).

## 6. Evaluation

1. **Representation quality:** per-dimension latent std, covariance
   off-diagonal magnitude, collapse flag (G1), latent prediction MSE.
2. **Dynamics quality:** horizon errors `[1,2,5,10,20]`, open-loop stability,
   action-conditioned vs unconditioned gap.
3. **Raw-state Linear Probe (fit on TRAIN ONLY):** fit a linear/ridge map
   `z_pred → x_{t+1}` using TRAIN representations with the trained encoder
   frozen; evaluate frozen_test and parameter_ood_test WITHOUT refitting.
   Report RMSE / MAE / `Skill = 1 − MSE_probe/MSE_persistence` against E0.
4. **Physical generalisation:** report by `(m, L, b)` groups and explicitly
   quantify Parameter-OOD degradation (G6 does not require high OOD skill, only
   an honest measurement).
5. **Statistics:** 5 seeds, episode-level paired bootstrap (10,000 resamples,
   95% CI), mean AND median. Correlated timesteps are never independent samples.

## 7. Acceptance gates (operational criteria — to be reviewed)

| Gate | Criterion |
|---|---|
| G0 | RK4 replay with the exact persisted action reproduces `x[t+1]`; manifest SHA-256 + parameter ranges + split isolation verified |
| G1 | No latent collapse: per-dim std above declared threshold, off-diagonal covariance bounded |
| G2 | E2 error < E1, paired 95% CI excludes 0 |
| G3 | E2 error < E3 shuffled-action, paired 95% CI excludes 0 |
| G4 | E2 Linear-Probe raw-state RMSE < E0 Persistence, skill > 0 |
| G5 | Horizons 1/2/5/10/20 finite, no divergence |
| G6 | Parameter OOD measured and degradation reported honestly (no pass/fail on magnitude) |

**Failed gates must be reported, never hidden.** A low latent prediction loss
alone is not evidence of learned physics, nor of planning/control ability.

## 8. Staged development order (task §6)

| Stage | Work | Acceptance |
|---|---|---|
| **P0** | Architecture audit + parameter-validation hardening | **This phase: audit DONE, fixes identified but NOT applied**; no regression to A1/A2/P2-02 |
| P0-fix | Apply the 3 validation fixes (§6 of audit) + tests | 210+ regression green; §8-banned files untouched |
| P1 | Controlled Pendulum simulator + dataset + audit script | G0 |
| P2 | Single-step `ActionConditionedWorldModel` + E0–E4 | G1–G3 |
| P3 | EMA + VICReg diagnostics (if not fully in P2) | no collapse |
| P4 | Multi-step JEPA + E5 | G4–G5 |
| P5 | Linear Probe + Parameter OOD | G6 |
| P6 | `P2_03A_FINAL_REPORT.md` + feature-branch commits | independent reproduction |

**P0 and P0-fix are separate commits.** Single-step JEPA must pass before any
multi-step loss is introduced (one variable at a time).

## 9. Proposed new files (all NET-NEW)

```
dynamics/trajectory_windows.py
models/action_conditioned_world_model.py
data/controlled_pendulum.py
configs/dynamics/controlled_pendulum_jepa.yaml
scripts/collect_controlled_pendulum.py
scripts/audit_controlled_pendulum.py
scripts/train_action_jepa.py
scripts/evaluate_action_jepa.py
tests/models/test_action_conditioned_world_model.py
tests/dynamics/test_trajectory_windows.py
datasets/controlled_pendulum_v1/{train,validation,frozen_test,parameter_ood_test}/manifest.json
docs/architecture/P2_03_ACTION_JEPA.md            (this file)
docs/architecture/P2_03A_FINAL_REPORT.md           (P6)
```

## 10. Restrictions & scientific boundaries

- Do NOT modify `models/world_model.py`, `models/predictor.py`,
  `models/conditional_predictor.py`, `utils/losses.py` behaviour,
  `train.py`, `data/pendulum.py`, existing datasets, or historical checkpoints.
- No `robotics` imports and no Pusher constants in generic `models/` / `dynamics/`.
- `pusher_v1` stays byte-identical; P2-02B contact-rich data is a **parallel,
  separate** effort and must not be mixed into this Core work.
- Frozen / Parameter-OOD splits never touch training, early stopping,
  hyper-parameter selection or Linear-Probe fitting.
- No MPC, no closed-loop control, no planning claims (that is P2-04).
- No claim of "physical law discovery" from a low latent loss.
- Commit to `feature/p2-03a-action-jepa` first; **do not auto-merge to `main`**.
- Full existing regression suite must remain green (baseline at `e4eccc0`:
  210 passed + 22 subtests).

## 11. Stop condition

Phase 0 delivered the audit, dependency map, gap analysis and this
specification. **No production code was modified.** Await review of the module
boundary, the exact window-index contract (§2.2), the E0/E1 definitions (Q1/Q2),
and the staged plan before P0-fix.
