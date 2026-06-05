# Transport-Aware Forecasting of Coastal Toxin Spikes — Design Spec

**Date:** 2026-06-02
**Branch context:** `oad-integration`
**Status:** Draft for review (brainstorming output → next step: writing-plans)
**Owner:** Anson Chen (ac283@uw.edu)

---

## 0. One-paragraph summary

An unsupervised satellite ocean-anomaly representation (OAD) validates against in-situ
biology at the offshore source (ESP mooring: *Pn* r=0.46, pDA r=0.33) but produces
**zero improvement** when naively appended as features to the per-site beach domoic-acid
(DA) forecaster (ΔR² = +0.0015, within seed noise). We argue the missing ingredient is
**transport**: the offshore anomaly must be advected to each beach with the correct,
physically-varying lag and direction before it carries predictive signal. We formalize
this as a **causal mediation problem** (offshore signal → transport mediator → beach DA),
introduce a **physics-constrained learned transport operator** that supplies the mediator,
and show it recovers offshore signal at the shore. The ambitious target is a **learned ocean
velocity field with end-to-end differentiable Lagrangian advection** (neural-ODE transport) —
the operator learns *how the water moves* and advects offshore bloom signal to each beach.
A **calibrated extreme-tail head** (conformal v1 → generative diffusion/flow nowcast stretch)
turns the forecast into operational spike alerts with coverage guarantees, and a
**leakage-aware spatiotemporal evaluation protocol** makes the comparison credible.

This is the **CS / methods paper** (Paper B). The applied DATect system paper (Paper A,
`paper/datect_paper_mdpi.tex`) is the downstream task it improves and cites.

---

## 1. Motivation and the central claim

### 1.1 The tension (already in the repo)

- **Offshore, the signal is real.** `docs/OAD_INTEGRATION_RESULTS.md` §16: OAD score correlates
  with ESP in-water measurements at NEMO (Pn r=+0.46 p=3e-5; pDA r=+0.33 p=1e-3).
- **At the beach, it vanishes.** Same doc §1: augmenting the ensemble with 16 OAD features
  yields ΔR² = +0.0015 (the model is marginally *better* without them; seed noise ±0.09).

### 1.2 The claim

The naive integration fails because it conditions a beach forecast on an **offshore** state
without modeling how that state is **physically delivered** to the beach. The offshore→shore
relationship is mediated by alongshore + cross-shelf transport, which in the PNW is governed
by upwelling/relaxation dynamics (bloom builds offshore during upwelling; relaxation events
deliver it shoreward — the Hickey/Trainer Juan-de-Fuca-eddy mechanism). A model that learns
this transport mediator should recover the signal the naive features waste.

### 1.3 Why this is novel (honest scoping)

- Neural operators (FNO), Lagrangian particle tracking (OceanParcels), and data assimilation
  exist in oceanography/ML, **but** the intersection — a learned, physically-conditioned
  transport operator as the **bridge from an unsupervised satellite ocean-state representation
  to sparse point toxin forecasting**, justified by **causal mediation** — is open.
- The causal-mediation framing is the intellectual spine and is essentially absent from HAB-ML.

---

## 2. Contributions (B-led)

1. **(B, headline) Physics-constrained learned transport operator** mapping the gridded
   offshore ocean-anomaly field to a per-site, transport-adjusted predictor, conditioned on
   BEUTI (upwelling-transport) and season. Constraints: causal (past-only), lag bounded by
   max physical current speed, alongshore-directional. **Crown-jewel target (L4):** a *learned
   ocean velocity field* with end-to-end differentiable Lagrangian particle advection
   (neural-ODE) — the model learns the transport dynamics rather than a fixed propagation kernel.
2. **(spine) Causal-mediation analysis** showing the offshore signal's effect on beach DA is
   mediated by transport — naive conditioning leaves the mediated path unmodeled; the operator
   recovers it. Quantify direct vs. mediated effect.
3. **(C, supporting) Calibrated extreme-tail head** producing distribution-free spike-alert
   thresholds with coverage guarantees, replacing the hand-tuned quantile clipping in
   `per_site_models.py`. **v1:** split/adaptive conformal. **Stretch:** a generative
   (diffusion/normalizing-flow) probabilistic nowcast head emitting the full predictive
   distribution, gated on the core landing first.
4. **(D, supporting) Leakage-aware spatiotemporal HAB-nowcasting benchmark**: centered-composite
   lag rule, leave-prior-years climatology, train/val/holdout windows, per-site + pooled
   metrics — formalized from `scripts/eval/`.

---

## 3. Architecture and approach

### 3.1 Ambition ladder (and the go/no-go gate)

| Level | What | Role |
|---|---|---|
| **L1** | Hand-built advection-lag feature: sample upstream-region OAD at lag = distance / speed(BEUTI) | **Week-1 go/no-go gate** + ablation baseline |
| **L2** | Learned 1-D alongshore transport operator (attention/conv kernel conditioned on BEUTI + season) emitting a per-site adjusted anomaly | Minimum publishable method |
| **L3** | End-to-end differentiable, physics-constrained transport operator over the 2-D coastal field → per-site latent → calibrated forecast head, trained against DA | Strong method |
| **L4** | **Crown jewel: learned ocean velocity field + differentiable Lagrangian advection (neural-ODE).** Parameterize a (season/BEUTI-conditioned) surface-velocity field `v(x, t; θ)`; advect offshore anomaly "particles" forward via a differentiable ODE solver to each site; train the velocity field end-to-end against beach DA. The model learns *how the water moves*, not just a propagation weight. | **Ambitious target** (you have time) |

**Gate rule:** If L1 cannot beat seed noise (|ΔR²| or Δspike-recall) on the validation window,
the offshore signal may be undeliverable; **pivot to a C-led paper** (calibrated spike
forecasting, which needs no OAD). Do not sink weeks into L2/L3/L4 before L1 clears.
**Climb the ladder in order** — each rung is a publishable result and de-risks the next; L4 is
attempted only after L3 (or at least L2) is working, so the timeline never rides solely on the
hardest version.

### 3.2 Spatial structure

10 sites form a 1-D alongshore chain (Kalaloch 47.6°N → Gold Beach 42.4°N). Transport is
dominantly alongshore + cross-shelf, so the operator is a 1-D advection model over this chain
(not a full 2-D PDE) — far more tractable. Each site maps to one OAD region
(`SITE_TO_REGION`). L3 optionally lifts to the 2-D field.

### 3.3 The transport operator (L2/L3)

- **Input:** OAD anomaly over regions × recent time window (causal, ending at the
  centered-composite-safe lag `test_date − 12`), plus BEUTI, season encoding, discharge.
- **Operator:** a learned propagation kernel `K(Δalongshore, Δt | BEUTI, season)` giving the
  weight with which upstream offshore anomaly at lag Δt contributes to site s. Implemented as
  attention over (region, time-lag) with the kernel modulated by BEUTI/season; or a small
  conv over the alongshore axis with learned, BEUTI-gated dilation.
- **Physical constraints:** (a) causal — only Δt ≥ minimum transit; (b) lag-bounded — support
  truncated at distance / min_current_speed; (c) non-negative kernel (mass-like);
  (d) direction flips with BEUTI sign (upwelling equatorward vs. relaxation onshore).
- **Output:** per-site transport-adjusted anomaly feature(s) → fed to the existing ensemble
  (L2) or to a differentiable forecast head (L3).

#### 3.3.1 L4 — learned velocity field + differentiable Lagrangian advection (crown jewel)

- **Velocity field:** a neural net `v(x, t) = f_θ(x, BEUTI_t, season_t, climate_t)` over the
  coastal domain (or the 1-D alongshore axis as a first cut), outputting a surface advection
  vector. Physically regularized: bounded magnitude (≤ max observed current speed), smoothness
  penalty, and a BEUTI-sign prior (equatorward/offshore during upwelling, onshore during
  relaxation).
- **Advection:** seed "particles" at the offshore anomaly locations (weighted by OAD intensity)
  and integrate `dx/dt = v(x, t)` forward with a **differentiable ODE solver** (`torchdiffeq`)
  over the causal window up to the leakage-safe lag. Particle mass arriving in each site's
  nearshore catchment becomes the transport-delivered predictor.
- **Training:** end-to-end against beach DA; gradients flow through the ODE solver into `v`.
  The learned field is itself a *result* — visualize it and check it against known PNW
  circulation (poleward Davidson current in winter, equatorward upwelling jet in summer) as a
  physical-plausibility validation.
- **Why it's the crown jewel:** it elevates the paper from "a learned feature" to "a learned
  dynamical transport model," which is a genuine methods contribution (neural-ODE / Lagrangian
  advection for EO→point forecasting) rather than an application of an off-the-shelf model.
- **Risk controls:** ODE-through-training is finicky (stiffness, solver cost). Mitigate with a
  short integration horizon, adjoint method for memory, and L3 as the always-available fallback
  if L4 won't converge in time.

### 3.4 Calibrated extreme-tail head (C)

- **v1 (conformal):** wrap forecast output with split/adaptive conformal prediction
  (ACI, Gibbs & Candès) to produce coverage-guaranteed intervals; derive spike-alert
  thresholds at the DA risk boundaries (5/20/40 µg/g). Compare coverage + spike-recall/F2
  vs. current quantile clipping.
- **Stretch (generative nowcast head):** replace the point+conformal-interval output with a
  **generative probabilistic nowcast** — a conditional diffusion or normalizing-flow head that
  emits the full predictive distribution of beach DA (and, via L4, the advected-field arrival),
  conditioned on the transport-operator output. Spike alerts read off the upper tail directly;
  conformal can still wrap it for coverage guarantees (conformalized generative intervals).
  GenCast-adjacent. **Gated:** built only after the core transport result (≥ L2 + conformal v1)
  is landed, so the paper never depends on the generative head converging.

### 3.5 Evaluation protocol (D)

- Reuse engine's leak-free rules; formalize: centered-composite lag, leave-prior-years DOY
  climatology, 3-window split (train ≤ anchor; val [2019,2022); holdout [2022,2024]).
- Metrics: R², MAE, spike recall/precision/F2 (regression-union), conformal coverage; per-site
  + pooled; **multi-seed (seeds 42–46) mean ± std** — single-seed deltas are not reportable
  (holdout R² std ≈ 0.15).

---

## 4. Causal-mediation formalization (spine)

- DAG: `OAD_offshore → Transport → DA_beach`, with confounders (season, BEUTI, climate indices).
- Estimand: decompose total effect of offshore anomaly on beach DA into **direct** (unmediated)
  and **indirect/mediated-by-transport** components (Imai et al. mediation; or front-door if
  transport is unobserved-but-recoverable).
- Empirical test: naive model captures ~direct path only (≈0); operator supplies the mediator
  and recovers the indirect path (ΔR² > noise). Report mediated effect size + CI.

---

## 5. Files to create / modify

| Path | Action | Purpose |
|---|---|---|
| `forecasting/transport_operator.py` | **new** | L1 features + L2/L3 operator |
| `forecasting/transport_neural_ode.py` | **new** | L4 learned velocity field + differentiable Lagrangian advection (`torchdiffeq`) |
| `forecasting/raw_data_processor.py` | modify | inject transport-adjusted features (insertion point already used by OAD cols) |
| `forecasting/conformal_head.py` | **new** | calibrated extreme-tail alerts (C, v1) |
| `forecasting/generative_head.py` | **new** | stretch: diffusion/flow probabilistic nowcast head |
| `per_site_models.py` | modify | route clipping through conformal head behind a flag |
| `scripts/eval/transport_ablation.py` | **new** | no-OAD vs naive-OAD vs L1 vs L2 vs L3 vs L4, multi-seed |
| `scripts/eval/causal_mediation.py` | **new** | mediation decomposition (spine) |
| `docs/benchmark/` | **new** | benchmark spec + frozen splits (D) |

All heavy runs on Hyak (`/gscratch/stf/ac283/...`); local is editing + L1 smoke only.

---

## 6. Risks and honest caveats

1. **Signal may be genuinely dead** — prior null is real. Mitigated by the L1 week-1 gate +
   the C-led pivot.
2. **BEUTI is a transport *proxy*** — no explicit wind-vector/current field in the dataset.
   Option: pull OSCAR/HF-radar surface currents (adds scope; future work if time-boxed).
3. **Single-mooring offshore validation** (NEMO) limits the causal claim's external validity.
4. **Seed sensitivity** — every headline number must be multi-seed mean ± std.
5. **Don't over-center OAD** — per the steer, OAD is "the input field"; the operator + forecast
   are the stars.

---

## 7. Sequence of work

1. **Week 1 (gate):** L1 hand-built advection feature; multi-seed val delta. Go/no-go.
2. L2 operator + ablation table (no-OAD / naive / L1 / L2), holdout.
3. C conformal head (v1) + coverage/spike-recall comparison.
4. D benchmark formalization + frozen splits.
5. Causal-mediation analysis (spine).
6. L3 differentiable/physics-constrained operator.
7. **L4 crown jewel:** neural-ODE velocity field + differentiable Lagrangian advection;
   physical-plausibility check of the learned field.
8. **Stretch:** generative (diffusion/flow) nowcast head, gated on ≥L2 + conformal landing.
9. ESP recompute on mae070 (also unblocks the OAD-paper checkpoint fix).

---

## 8. Relationship to the other papers

- **Paper A (applied, `paper/`):** DATect system; the downstream task. Submit to MDPI Toxins /
  Harmful Algae. Largely done.
- **OAD paper (`paper_oad/`, `paper_oad_cvpr/`):** the representation; checkpoint headline being
  corrected mae050 → **mae070** (forecastability + climatology verified; ESP recompute pending).
- **This paper (B):** cites A as downstream task and the OAD paper as the input representation;
  the transport operator is the new contribution that connects them.

---

## 9. Resolved decisions

- **Ambition (resolved):** climb L1→L2→L3→**L4 (neural-ODE velocity field, crown jewel)**;
  L4 is the headline target given ample time, with L2/L3 as always-available fallbacks.
- **Output head (resolved):** conformal v1, **generative diffusion/flow nowcast as a gated
  stretch** (built only after the core lands).
- **External surface currents (resolved): OUT for v1** — BEUTI proxy only; OSCAR/HF-radar is v2/roadmap.
- **Multi-task Pn+DA (resolved): OUT** — roadmap only (not selected).
- **Venue (resolved):** attempt NeurIPS/ICLR main track; fallbacks = climate-ML workshop /
  JAMES / Environmental Data Science.

## 10. Roadmap / explicitly out of v1 scope

- External real surface-current fields (OSCAR / HF-radar) to drive/validate transport.
- Full 2-D neural operator (FNO) over the coastal field.
- Joint *Pn* (organism) + DA (toxin) multi-task forecasting.
- Transferable backbone / few-shot transfer to new sites (conflicts with "OAD stays small").
