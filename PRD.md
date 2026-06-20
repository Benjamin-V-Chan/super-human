# Prosthesis-RL — PRD

**Prosthesis-RL** is an AI-driven system that designs and optimizes custom robotic
prosthetic arms from a short patient video. It runs a closed **perceive → design → verify →
optimize → manufacture** loop: a vision agent extracts the patient's pain points and
constraints, a reasoning agent proposes arm parameters, a MuJoCo verifier grades the design
on real ADL tasks, an RL loop (GRPO) trains the designer on that feedback, and the winning
design is exported as a printable STL.

➡️ **The full, canonical PRD lives in [`docs/PRD.md`](docs/PRD.md).**

For team ownership and the hour-by-hour timeline, see [`WORK_SPLIT.md`](WORK_SPLIT.md).
