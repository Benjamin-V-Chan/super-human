# Prosthesis-RL — Product Requirements Document

> **Status:** Draft · **Last updated:** 2026-06-20 · **Owner:** Vihaan Shringi (lead)
> **Companion doc:** [`WORK_SPLIT.md`](../WORK_SPLIT.md) — who owns which slice.
> This is the **canonical, unified PRD**. It consolidates the full product into one place;
> the work-split doc covers ownership and timeline detail.

---

## 1. Overview / Vision

**Prosthesis-RL** is an AI-driven system that designs and optimizes **custom robotic
prosthetic arms** from a short video of a patient. Instead of a slow, manual,
one-size-fits-all fitting process, Prosthesis-RL runs an automated
**design → verify → learn** loop:

1. **Perceive** the patient's pain points and physical constraints from video.
2. **Design** an arm specification tailored to their Activities of Daily Living (ADL).
3. **Verify** that design in a physics simulator against concrete tasks.
4. **Optimize** the designer through reinforcement learning on the verifier's feedback.
5. **Manufacture** the winning design as a printable CAD/STL file.

The core innovation is the **closed loop**: orchestrated AI agents that learn from a
physics verifier rather than guessing, so each generation of designs is measurably better
than the last on the patient's real tasks.

---

## 2. Problem & Users

**Primary user:** People with upper-limb loss who need a prosthetic arm tuned to *their*
body (residual limb strength, range of motion, grip capacity) and *their* daily tasks
(reaching, grasping, self-feeding).

**Today's problem:**
- Prosthetic design is manual, slow, and expensive, and rarely personalized to a
  patient's specific tasks and constraints.
- There is no fast, automated way to translate "what this patient struggles with" into a
  validated, manufacturable arm specification.

**Our bet:** A perception agent can extract a structured problem statement from video, a
design agent can propose arm parameters, and a fast physics verifier can score those
designs well enough that an RL loop converges on a genuinely good, personalized arm — end
to end, automatically.

---

## 3. Goals & Non-Goals

### Goals (hackathon scope)
- Close a **real end-to-end loop**: video → `ProblemSpec` → `DesignParams` → simulated
  grade → reward → improved design → exported STL.
- Make the simulation **faithful enough** that a design winning in-sim would plausibly
  work on a real arm.
- Train the design agent with RL so it **learns from the verifier** instead of guessing.
- Produce a compelling demo: a personalized arm designed from a real patient video, plus
  an STL and a LeRobot demonstration.

### Non-Goals (for this build)
- Clinical validation, regulatory approval, or real-patient deployment.
- Physical manufacturing/printing beyond producing the STL file.
- A learned low-level control policy is a **stretch**, not a requirement (scripted IK is
  acceptable for v1).
- Production-grade web/front-end; no UI framework is committed at this stage.

---

## 4. Success Metrics

| Metric | Target |
|--------|--------|
| End-to-end eval runs | `hud eval tasks.py claude` returns a **real number** (by Sat dinner) |
| Task difficulty calibration | Each task lands at **20–50% mean reward with real variance** |
| Loop closure | Dumb end-to-end loop with stubs closes **Saturday night**; real components swapped in after |
| RL kickoff | First GRPO training run kicked off by **8 AM Sunday** (hard deadline) |
| Deliverable | STL export of the top design + LeRobot demo by **Sun 1 PM** |
| Determinism | Verifier reward is a **single deterministic scalar** per episode (reproducible) |

---

## 5. System Architecture

Five stages, connected by three shared contracts. The same contracts let each slice be
stubbed first and swapped for the real implementation later.

```
   patient video
        │
        ▼
┌──────────────────┐   ProblemSpec    ┌──────────────────┐   DesignParams   ┌──────────────────┐
│  PERCEIVE        │ ───────────────▶ │  DESIGN          │ ───────────────▶ │  VERIFY          │
│  Claude vision   │                  │  Claude reasoning│                  │  MuJoCo + grading│
│  (CV backend)    │ ◀─────────────── │  agent           │ ◀─────────────── │  (parametric arm)│
└──────────────────┘   sim feedback   └──────────────────┘   reward scalar  └──────────────────┘
                                              ▲                                       │
                                              │ updated policy                        │ DesignParams
                                  ┌──────────────────┐                     ┌──────────────────┐
                                  │  OPTIMIZE        │                     │  MANUFACTURE     │
                                  │  GRPO / Fireworks│                     │  CadQuery/OpenSCAD│
                                  │  (HUD platform)  │                     │  → STL (Daytona) │
                                  └──────────────────┘                     └──────────────────┘
```

- **Perceive:** clip → frame extraction → VLM (Claude vision) → pain-point detection →
  emits a validated `ProblemSpec`.
- **Design:** consumes `ProblemSpec` + last sim feedback → emits `DesignParams` + control
  hints. The agent runs inside an action/observation interface that RL trains against.
- **Verify:** builds a parametric MuJoCo arm from `DesignParams`, runs ADL task scenes,
  grades them into a single reward scalar. Must be **deterministic and fast** (called
  thousands of times).
- **Optimize:** GRPO rolls out the design agent ~10×/task and trains on trajectories where
  good designs were rewarded.
- **Manufacture:** converts the winning `DesignParams` into an STL via CadQuery/OpenSCAD,
  executed inside a Daytona sandbox.

The orchestration that wires these stages together — plus the HUD integration — is the
"own the loop" slice. **If the loop doesn't close, nothing else matters**, so the loop is
stood up first with every component stubbed, then real pieces are swapped in.

---

## 6. Data Contracts  *(linchpin — lock by Saturday 2 PM)*

These three contracts are the **only things every slice must agree on**. They decouple the
slices so they can develop in parallel.

### `ProblemSpec`  — output of Perceive, input to Design
```json
{
  "tasks": ["reach", "grasp", "feeding"],
  "constraints": {
    "rom": "...",                 // patient range-of-motion limits
    "residual_strength": "...",   // residual limb strength
    "grip_capacity": "..."        // achievable grip
  }
}
```

### `DesignParams`  — output of Design, input to Verify & Manufacture
```json
{
  "upper_arm_len": 0.0,
  "forearm_len": 0.0,
  "joint_stiffness": 0.0,
  "grip_width": 0.0,
  "joint_limits": { "...": "..." }
}
```

### `Reward`  — output of Verify, training signal for Optimize
- A **single deterministic scalar per episode**, computed inside Nathan's verifier.
- Same inputs → same reward, every time (reproducibility is a hard requirement).

> Exact units, ranges, and field types are TBD and must be pinned down when the contracts
> are locked (see Open Questions).

---

## 7. Functional Requirements by Component

### 7.1 Perception agent + HUD task registry — *Vihaan*
- **Perception agent:** clip → VLM (Claude vision) → `ProblemSpec` JSON, with a **strict,
  validated schema** on the output.
- **HUD task registry (`tasks.py`):** register the ADL tasks, wire the gateway, and make
  `hud eval tasks.py claude` run end to end.
- **Close the dumb end-to-end loop** with every component stubbed by Saturday night.

### 7.2 Design agent scaffold — *Vihaan*
- Takes `ProblemSpec` + last sim feedback → emits `DesignParams` + control hints.
- Builds the **action/observation interface**; Vasi trains the policy inside it.

### 7.3 CAD bridge — *Benji*
- `DesignParams` → CadQuery/OpenSCAD → **STL**, executed inside the **Daytona** sandbox.

### 7.4 CV backend + physics verifier — *Nathan*
- **CV/perception backend:** frame extraction → VLM call → pain-point detection; runs on
  Modal if heavy. Produces the `ProblemSpec` the agent consumes.
- **MuJoCo environment + grading:** parametric arm model (XML generated from
  `DesignParams`), ADL task scenes, and grading functions —
  **reach success, grasp-force window, energy, ROM violation, self-collision**.
- Must be **deterministic and fast** (called thousands of times per training run).
- **Stretch:** promote one task into Antim Worldsim/Newton for a fidelity story.

### 7.5 RL + reward optimization — *Vasi*
- **Inner controller:** scripted/IK controller first (so reward reflects *design quality*,
  not control noise); upgrade to a learned policy only if time allows.
- **Reward shaping** (see §8); tune so every task lands at **20–50% mean reward** with real
  variance.
- **GRPO loop (Fireworks/HUD):** roll out the design agent ~10×/task; train on trajectories
  where good designs got rewarded.

---

## 8. Reward / Grading Specification

The verifier composes the per-episode reward scalar from the grading functions:

```
reward = success − energy − ROM_violation − collision    (weighted per task tier)
```

- **success** — task completed (reach reached, grasp held within force window, feeding
  motion achieved).
- **energy** — penalize energy expended (efficiency).
- **ROM_violation** — penalize exceeding the patient's range-of-motion constraints.
- **collision** — penalize self-collision.

Weights are tuned **per tier** so each task sits in the 20–50% mean-reward band with
genuine variance (not all-pass or all-fail). Exact per-tier weights are TBD (see Open
Questions).

---

## 9. Tech Stack, Infrastructure & API Budget

| Provider | Role | Budget / Notes |
|----------|------|----------------|
| **Anthropic (Claude)** | Vision (perception) + reasoning (design) — primary VLM | Primary |
| **HUD** | Env + eval API; RL training/eval platform | Core |
| **Modal** | GPU for CV backend + inference serving | **$250** |
| **Fireworks AI** | GRPO RL training | **$30** (unblock 8 AM Sun kickoff) |
| **Daytona** | CAD sandbox execution (Benji's CAD agents) | Core |
| **OpenAI** | Vision backup if Claude bottlenecks | Stretch/backup |
| **Antim Labs** | Worldsim/Newton + physical validation | Stretch |
| **Google DeepMind / GCP** | Bigger RL run if needed | Stretch |

Backup keys are kept **warm for instant swap-in**.

**Languages:** Python (agents, CV, RL, sim glue); JSON (schema validation for `ProblemSpec`
and `DesignParams`).

---

## 10. Milestones & Timeline

| Window | V + B (loop/API/CAD) | Nathan (CV + sim) | Vasi (RL) |
|--------|----------------------|-------------------|-----------|
| **Sat 12:30–7 PM** | Schema + agent scaffold + `tasks.py` + CAD bridge | Arm XML + Reach task + grading | Scripted IK + reward v1 |
| **Sat 7 PM–Sun 8 AM** | Real CV → real `ProblemSpec`; 1 personalized task live | Add grasp + feeding, tune 20–50% | GRPO config; **kick off run by 8 AM** |
| **Sun 8 AM–1 PM** | Loop video; converge | STL export; LeRobot demo | Pick top design; final eval |

**Hard deadlines:** GRPO run kicked off by **8 AM Sun** · **Sun 1 PM submission** ·
**2:30 PM top-10 presentation**.

Key checkpoints:
- **Sat 2 PM** — shared contracts locked.
- **Sat dinner** — `hud eval tasks.py claude` returns a real number; Reach task + grading
  callable by the design agent.
- **Sat night** — dumb end-to-end loop closed with stubs.

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Loop doesn't close | Project fails regardless of component quality | Stand up stubbed end-to-end loop **first** (Sat night), swap in real pieces after |
| Sim not faithful enough | In-sim winners don't transfer to real arm | Keep grading physically grounded; stretch to Antim Worldsim/Newton for fidelity |
| Reward has no variance | RL can't learn (all-pass / all-fail) | Tune weights to **20–50% mean reward with real variance** per task |
| Credit exhaustion | Training/inference blocked mid-run | Budget per provider ($250 Modal, $30 Fireworks); warm backup keys |
| Claude bottleneck | Perception/design stalls | OpenAI vision backup, keys warm for instant swap |
| Contracts churn | Slices block each other | **Lock `ProblemSpec` / `DesignParams` / reward by Sat 2 PM** |
| Control noise pollutes reward | Reward reflects control, not design | Scripted IK first; learned policy only as stretch |

---

## 12. Open Questions

- Exact field types/units/ranges for `ProblemSpec.constraints` (`rom`, `residual_strength`,
  `grip_capacity`) and `DesignParams.joint_limits`.
- Per-tier weights for the reward terms (`success`, `energy`, `ROM_violation`, `collision`).
- Definition of the grasp **force window** and the precise success criteria per ADL task.
- Which single task (if any) gets promoted into Antim Worldsim/Newton.
- Whether a learned inner-control policy is attempted, or scripted IK ships for the demo.

---

*Ownership and the hour-by-hour split live in [`WORK_SPLIT.md`](../WORK_SPLIT.md).*
