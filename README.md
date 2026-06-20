# Prosthesis-RL

RL-driven prosthesis design loop for generating, simulating, and improving personalized arm designs from ADL task observations.

This README is the working ownership map for the sprint. It is a companion to the PRD: each person owns a vertical slice, and the shared contracts between slices are the only interfaces everyone must agree on.

**Lock the shared contracts by Saturday 2 PM.**

## Core Interfaces

The system is split into three stages connected by strict contracts:

```ts
type ProblemSpec = {
  tasks: unknown[];
  constraints: {
    rom: unknown;
    residual_strength: unknown;
    grip_capacity: unknown;
  };
};

type DesignParams = {
  upper_arm_len: number;
  forearm_len: number;
  joint_stiffness: number;
  grip_width: number;
  joint_limits: unknown;
};

type Reward = number;
```

- `ProblemSpec`: emitted by perception and consumed by the design agent.
- `DesignParams`: emitted by the design agent and consumed by CAD + simulation.
- `Reward`: a single deterministic scalar per episode, computed by the verifier.

## File Outline

```text
.
├── README.md
├── pyproject.toml
├── tasks.py                         # HUD eval entrypoint
├── prosthesis_rl/
│   ├── agents/
│   │   ├── orchestrator.py          # End-to-end loop owner
│   │   ├── perception.py            # Clip -> ProblemSpec
│   │   └── design.py                # ProblemSpec -> DesignParams
│   ├── contracts/
│   │   └── schemas.py               # ProblemSpec, DesignParams, Reward contracts
│   ├── cv/
│   │   └── backend.py               # Frame extraction + VLM backend placeholder
│   ├── cad/
│   │   └── bridge.py                # DesignParams -> STL bridge
│   ├── sim/
│   │   ├── mujoco_env.py            # Parametric ADL sim placeholder
│   │   └── verifier.py              # Deterministic reward computation
│   ├── rl/
│   │   ├── controller.py            # Scripted IK first pass
│   │   ├── rewards.py               # Reward shaping helpers
│   │   └── train.py                 # GRPO training stub
│   ├── hud/
│   │   └── gateway.py               # HUD-facing eval gateway
│   └── config/
│       └── defaults.py
├── scripts/
│   └── run_loop.py                  # Local smoke run
├── tests/
│   └── test_smoke_loop.py
├── examples/
│   └── adl/
└── assets/
    ├── clips/
    └── stl/
```

## Ownership

### Vihaan + Benji: Agent Architecture + API Infra

Own the orchestration loop across all three stages, plus HUD integration.

If the loop does not close, nothing else matters. The goal is to get a dumb end-to-end loop running by Saturday night with every component stubbed, then let Nathan and Vasi swap in real pieces.

#### Deliverables

- Perception agent: clip -> VLM using Claude vision -> validated `ProblemSpec` JSON.
- Design agent scaffold: consumes `ProblemSpec` plus last sim feedback, emits `DesignParams` plus control hints.
- Action/observation interface for Vasi's policy.
- HUD task registry in `tasks.py`: register ADL tasks, wire the gateway, and make `hud eval tasks.py claude` run end to end.
- CAD bridge: `DesignParams` -> CadQuery/OpenSCAD -> STL, executed inside the Daytona sandbox.

#### Credits

- HUD: environment and eval API.
- Anthropic: Claude vision and reasoning.
- Modal: inference serving if needed.

#### Milestone

`hud eval tasks.py claude` returns a real number by dinner Saturday.

### Nathan: CV + Physics Sim

Own the verifier.

The sim is the heart of the project. It should be faithful enough that a design winning in the environment would plausibly work on a real arm.

#### Deliverables

- CV/perception backend: frame extraction -> VLM call -> pain-point detection.
- Produce the `ProblemSpec` consumed by the agent.
- Run CV on Modal if heavy.
- MuJoCo environment and grading:
  - Parametric arm model with XML generated from `DesignParams`.
  - ADL task scenes.
  - Grading functions for reach success, grasp force window, energy, ROM violation, and self-collision.
- Deterministic, fast verifier suitable for thousands of calls.
- Stretch: promote one task into Antim Worldsim/Newton for a fidelity story.

#### Credits

- Modal: $250 GPU budget for CV.
- Antim Labs: Worldsim and physical validation.
- OpenAI: vision backup if Claude bottlenecks.

#### Milestone

Reach task `1.1` plus grading callable by the design agent by dinner Saturday.

### Vasi: RL + Optimization

Own training the design agent and shaping the reward.

The goal is to make the designer learn from the verifier instead of guessing.

#### Deliverables

- Inner controller: scripted/IK controller first, so reward reflects design quality rather than control noise.
- Upgrade to a learned policy only if time allows.
- Reward shaping:
  - success
  - minus energy
  - minus ROM violation
  - minus collision
- Weight rewards per tier from PRD section 7.
- Tune so every task lands at 20-50% mean reward with real variance.
- GRPO loop using Fireworks/HUD:
  - Roll out the design agent roughly 10 times per task.
  - Train on trajectories where good designs received reward.

#### Credits

- Fireworks AI: $30 RL training.
- HUD: platform training and evals.
- Google DeepMind: only if GCP is needed for a bigger run.

#### Milestone

First training run kicked off by 8 AM Sunday. This is a hard deadline.

## Timeline

| Window | Vihaan + Benji | Nathan | Vasi |
| --- | --- | --- | --- |
| Sat 12:30-7 PM | Schema, agent scaffold, `tasks.py`, CAD bridge | Arm XML, Reach task, grading | Scripted IK, reward v1 |
| Sat 7 PM-Sun 8 AM | Real CV -> real `ProblemSpec`; one personalized task live | Add grasp + feeding, tune to 20-50% | GRPO config; kick off run by 8 AM |
| Sun 8 AM-1 PM | Loop video; converge | STL export; LeRobot demo | Pick top design; final eval |

## Submission

- Sunday 1 PM: submission deadline.
- Sunday 2:30 PM: top-10 presentation.
