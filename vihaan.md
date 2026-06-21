# Vihaan — Context & Task List

My working doc for the Prosthesis-RL hackathon. See [`docs/PRD.md`](docs/PRD.md)
for the full product spec and [`WORK_SPLIT.md`](WORK_SPLIT.md) for who owns what.

---

## Context (what we're building)

**Prosthesis-RL** designs custom prosthetic arms from a short patient video. It
runs a closed loop:

```
patient video → ProblemSpec → DesignParams → simulated grade → reward → better design → STL
   (perceive)     (design)       (verify)      (optimize)        (manufacture)
```

**My slice (Vihaan):** own the loop orchestration + perception + HUD eval, plus
provision/max every sponsor API. **Benji** took the CAD agents.

**Team:**
- **Vihaan (me)** — loop, perception, HUD, API maxxing
- **Benji** — CAD agents (`DesignParams` → CadQuery/OpenSCAD → STL in Daytona)
- **Nathan** — CV backend + MuJoCo physics verifier (the reward scalar)
- **Vasi** — RL/GRPO + reward shaping (consumes my action/obs interface)

**Shared contracts** (frozen, in `prosthesis_rl/contracts/schemas.py`):
- `ProblemSpec` — perceive → design
- `DesignParams` — design → verify/manufacture
- `Reward` — single deterministic scalar per episode, from Nathan's verifier

**CV decision:** perception/problem-detection uses **Gemma video analysis**
(Google GenAI) — extract frames → Gemma multimodal → detect struggled ADL tasks
+ physical constraints → map to `ProblemSpec`. Falls back to a deterministic
detection with no key so the loop never breaks.

---

## My tasks

### ✅ Done / built
- **Loop orchestration** — `ProsthesisLoop` closes perceive→design→cad→verify;
  `scripts/run_loop.py` returns a real reward scalar.
- **HUD `tasks.py`** — `claude()` entrypoint returns a real number; target
  `hud eval tasks.py claude`. HUD CLI installed, `HUD_API_KEY` set.
- **Task #1 — Perception pipeline (Gemma)** — `clip → frames → Gemma → ProblemSpec`:
  - `cv/frames.py` (ffmpeg frame sampling)
  - `cv/gemma.py` (`GemmaVideoAnalyzer`, JSON detection + fallback)
  - `cv/backend.py` (detection → validated `ProblemSpec`)
  - `agents/perception.py` (wired to backend)
  - `scripts/perception_demo.py` + `tests/test_perception_pipeline.py` (6 tests green)

### 🔜 To do — build (coding)
| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Perception agent (Gemma → `ProblemSpec`) | **built, fallback path** | activate live by setting `GOOGLE_API_KEY` / `GEMINI_API_KEY` |
| 2 | Design agent scaffold + action/obs interface | pending | **Vasi blocked on this** — hand off ASAP |
| 3 | HUD `tasks.py` end-to-end | **stub passing** | swap stub verifier for Nathan's MuJoCo |
| 4 | Close the dumb loop (all stubbed) | **done in stub** | real pieces swap in per slice |

### 🔜 To do — API maxxing (provision first, then wire)
| # | Sponsor | Status | Notes |
|---|---------|--------|-------|
| 5 | Anthropic (Claude vision + reasoning) | pending | design reasoning; Claude is design-side, Gemma is CV-side |
| 6 | HUD (env + eval + training) | **key set** | `hud init` still needs a name/preset or interactive run |
| 7 | Modal ($250 GPU) | pending | Nathan's CV backend / inference serving |
| 8 | Fireworks AI ($30 GRPO) | pending | **8 AM Sun** hard deadline for Vasi's kickoff |
| 9 | Daytona (CAD sandbox) | pending | Benji's CAD agents execute inside it |
| 10 | Backups: OpenAI / Antim / GDM | pending | keys warm, swap-in instant |

---

## Priority order (right now)

1. Lock contracts (done — frozen in `schemas.py`)
2. **Provision keys:** Gemma/Google (activates #1), Anthropic (#5), HUD (done)
3. **Design scaffold + action/obs interface (#2)** — unblocks Vasi
4. **Fireworks (#8)** — provision tonight, 8 AM Sun gate
5. Swap real pieces into the stubbed loop (Nathan's sim → #3/#4)
6. Modal (#7), Daytona (#9), backups (#10)

---

## How to run

```bash
# venv (Python 3.12 — hud-python needs <3.13)
.venv/bin/python -m pip install -e ".[dev,cv]"

# perception pipeline demo
PYTHONPATH=. .venv/bin/python scripts/perception_demo.py examples/adl/reach_1_1.mp4

# full loop + HUD entrypoint
PYTHONPATH=. .venv/bin/python scripts/run_loop.py
PYTHONPATH=. .venv/bin/python tasks.py

# tests
PYTHONPATH=. .venv/bin/python -m pytest -q

# activate live Gemma
export GOOGLE_API_KEY=...   # or GEMINI_API_KEY
```
