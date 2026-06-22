# Super Human — Product Requirements Document

> **Event:** YC x HUD Hackathon
> **Status:** Active build
> **Pivot:** Simulation-first. The deliverable is a reproducible sim bundle — morphology, environment, policy, evaluation report, rollout video. It is not a physical prosthesis, medical device, or manufacturing workflow.
> **Related docs:** [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md), [WORK_SPLIT.md](WORK_SPLIT.md)

---

## 1. Overview

**Super Human** is a simulation-first system for rapidly designing and validating personalized assistive-limb morphologies. Given a short egocentric clip or a structured task description, it:

1. Runs a vision model (Gemini) to extract a `TaskSpec` — the action, affected side, grip pattern, range-of-motion, and limb measurements.
2. Drives an LLM-based design agent to generate explicit `DesignParams` candidates (links, joints, limits, actuators, masses) and rank them through spatial validation gates.
3. Assembles a MuJoCo scene from the generated morphology, runs ≥ 10 fixed-seed rollouts with a scripted or learned policy, and scores each candidate.
4. Exports evaluation metrics, a rollout video, and CAD geometry (STL/MJCF) for the best design.

The product outcome is a **reproducible sim bundle**: scenario config, morphology, policy artifact, evaluation report, and replay video.

---

## 2. Motivation

Prosthetics are not one-size-fits-all. Every user has a different injury, body geometry, and daily routine — but the clinical workflow for matching a device to a person's actual tasks is slow, manual, and expensive. Super Human bets that AI design reasoning combined with physics simulation can search the morphology space and validate a candidate design fast, producing measurable evidence *before* any real hardware exists.

---

## 3. Users

**Primary:** a researcher, engineer, or evaluator who wants to test whether a simulated assistive limb can complete a concrete ADL task for a specific person.

**Secondary:** a team member comparing morphology or controller choices using reproducible reward data, rollout videos, and failure-mode analysis.

---

## 4. Goals

- Convert a task clip or structured prompt into a concrete simulation scenario via `TaskSpec`.
- Generate candidate simulated limb morphologies with valid kinematics, joint limits, masses, and collision geometry, validated by spatial gates before sim entry.
- Run a scripted IK controller as a baseline; support RL training when the scenario and reward are stable.
- Evaluate candidates deterministically: fixed seeds, repeated rollouts, comparable metrics across designs.
- Produce a final sim package: environment config, policy checkpoint, metrics, and demo video.

---

## 5. Non-Goals

- Physical prosthesis delivery or manufacturing.
- Clinical validation or medical claims of any kind.
- Hardware-ready CAD — geometry exists only as simulation input.
- Human-subject deployment or regulatory approval.

---

## 6. End-to-End Flow

```
egocentric clip or task prompt
        │
        ▼
   perception (Gemini vision model)
        │  TaskSpec: action, affected side, grip type, ROM, measurements
        ▼
   design agent (LLM + spatial gates)
        │  DesignParams: links, joints, limits, actuators, masses
        │  validation: kinematic tree, reachability, inertia, actuator ranges
        ▼
   simulation assembly (MuJoCo / HUD)
        │  SimSpec: scene, physics hz, seeds, reward terms, obs space
        ▼
   policy / controller
        │  scripted IK baseline → optional RL (PPO / SAC)
        ▼
   evaluation
        │  success rate, mean reward, energy, collision rate, joint-limit violations
        ▼
   artifacts
        rollout video, metrics report, STL/MJCF export, policy checkpoint
```

---

## 7. Core Requirements

### 7.1 Perception

- Accept a short egocentric clip, existing ADL clip, or structured task prompt.
- Produce a `TaskSpec` with: task goal, object list, success condition, observation requirements, and explicit assumptions for uncertain clip details.
- Do not overclaim patient-specific medical inference — uncertain measurements become annotated assumptions, not hard values.

### 7.2 Simulation Assembly

- Convert `TaskSpec` into a `SimSpec` defining scene geometry, object poses, reward terms, initial state distribution, and episode length.
- Generate or configure a `DesignParams` morphology and export it as a valid MJCF model.
- Validate the model before any policy runs:
  - Kinematic tree is acyclic and all joints have valid axes.
  - Masses and inertias are physically plausible.
  - Actuator torque limits cover the expected task loads.
  - Reachable workspace covers the target region.
  - Collision geometry does not cause simulator crashes at reset.

### 7.3 Policy and Control

- Ship a scripted or IK policy as the baseline so the demo has a reliable floor.
- Support RL training (PPO or SAC) when the scenario and reward produce useful variance.
- Export the controller as a `PolicyArtifact`: either a `.pt` checkpoint or a runnable JSON config.

### 7.4 Evaluation

- Run ≥ 10 rollouts per candidate with fixed random seeds for determinism.
- Report: task success rate, mean/variance of episode reward, mean energy, collision rate, joint-limit violation rate, and stability (no sim crashes).
- Save a rollout video and optionally a web-viewer replay for the final demo.

---

## 8. Success Metrics

| Metric | Target |
|---|---|
| Runnable sim | Task environment launches locally or through HUD without crashes |
| Policy artifact | Scripted policy or `.pt` checkpoint loads and runs without errors |
| Determinism | Fixed-seed rollouts produce bit-identical metrics across runs |
| Task success | Final demo completes at least one clear ADL-style task in sim |
| Eval evidence | Report includes success rate, reward distribution, failure modes, rollout video |
| Scope clarity | All public docs make clear the product is simulation-only |

---

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Sim task is too vague | Policy cannot be evaluated cleanly | Define explicit success condition and termination criteria before training |
| Morphology invalid | MuJoCo cannot load the environment | Validation gates reject bad models before sim entry |
| Reward has no useful variance | RL cannot improve | Tune difficulty with baseline rollouts before running RL |
| Clip interpretation is uncertain | Scenario becomes speculative | Record uncertain values as explicit assumptions in `TaskSpec` |
| Demo depends on training instability | Final demo may fail | Keep scripted/IK baseline always runnable |

---

## 10. Final Deliverable

The hackathon deliverable is a sim bundle containing:

- Task/scenario config (`SimSpec` + scene assets).
- Generated assistive-limb morphology (`DesignParams` + MJCF).
- Controller or policy artifact (`.pt` or runnable config).
- Evaluation report (success rate, reward, failure modes).
- Rollout video or web-viewer replay.

Any physical or clinical interpretation is explicitly out of scope.
