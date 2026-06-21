# Prosthesis-RL SIM

Prosthesis-RL SIM is a simulation-first research project for designing and evaluating assistive-limb behavior in virtual daily-living tasks.

The project focuses on a reproducible loop:

```text
task description -> simulated environment -> policy rollout -> evaluation
```

The output is a runnable simulation demo with metrics and replay evidence. It is not a physical prosthesis, medical device, or manufacturing workflow.

## Docs

- [Product spec](docs/PRD.md)
- [Technical plan](docs/TECHNICAL_PLAN.md)
- [Work split](docs/WORK_SPLIT.md)

## Local Viewer

```bash
cd viewer
cp .env.example .env
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173).
