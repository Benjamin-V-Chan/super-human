# Prosthesis-RL

RL-driven prosthesis design loop for generating, simulating, evaluating, and improving personalized arm designs from Activities of Daily Living (ADL) observations.

Prosthesis-RL turns a short patient video into a structured problem statement, proposes a prosthetic-arm design, verifies that design in simulation, improves future designs from the verifier's feedback, and exports the winning design as a printable CAD/STL artifact.

For the detailed product spec, see [docs/PRD.md](docs/PRD.md). For team responsibilities, milestones, and task ownership, see [WORK_SPLIT.md](WORK_SPLIT.md). AI-agent coordination, path ownership, handoffs, and merge gates are defined in [AGENTS.md](AGENTS.md).

## System Loop

```text
patient video
    |
    v
Perceive -> Design -> Verify -> Optimize
              |          ^
              v          |
          Manufacture    |
```

- **Perceive:** extract patient constraints, target ADL tasks, and relevant context into a validated `ProblemSpec`.
- **Design:** use AI reasoning to propose prosthesis parameters as `DesignParams`.
- **Verify:** simulate the design against ADL tasks and produce a deterministic reward.
- **Optimize:** improve design proposals from empirical verifier feedback.
- **Manufacture:** convert the winning design into a CAD/STL output.

## Core Interfaces

The project is organized around three contracts:

```ts
type ProblemSpec = {
  tasks: string[];
  constraints: {
    rom: unknown;
    residual_strength: number;
    grip_capacity: number;
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
- `DesignParams`: emitted by the design agent and consumed by CAD, simulation, and evaluation.
- `Reward`: a deterministic scalar score from the verifier.

## Repository Outline

```text
.
├── README.md
├── PRD.md
├── WORK_SPLIT.md
├── docs/
│   └── PRD.md
├── adl/
│   └── README.md
└── viewer/
    ├── server.js
    ├── src/
    └── package.json
```

## Local Viewer

```bash
cd viewer
cp .env.example .env
# add ANTHROPIC_API_KEY to .env
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173).

## Scope

This is a research and demo system, not a medical device. Any real prosthetic or assistive-limb deployment would require biomechanical safety review, clinical validation, and regulatory approval.
