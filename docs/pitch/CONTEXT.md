# ARMASAI — Pitch Context

> Single source of truth for the pitch deck. The `/pitch-deck` command reads this
> file (plus the linked repo docs) and builds the slides from it. Edit here, not
> in the deck.

## One-liner

**ARMASAI turns a pair of Meta Ray-Ban glasses into a personalized prosthetics
engineer** — it watches your daily routine, finds the tasks you struggle with,
and designs a prosthetic tailored to *your* life, not an average body.

## Origin story (the human hook — Problem slide)

Our friend **Benji** was working out when he tore a muscle so badly it triggered
**rhabdomyolysis** (muscle breakdown that floods the bloodstream and can shut
down the kidneys). He ended up in the **ER**. The muscle is torn open and no
longer works — and now everyday actions he never thought about have become hard:
opening bottles, carrying things, the small one-handed tasks that make up a day.

**Photo to use on the Problem slide:** `assets/pitch/benjihospital.jpg`
(Benji in the ER). Treat it with respect — it's a real photo of our friend; this
is why we built ARMASAI.

## The problem (general)

- Prosthetics are **not one-size-fits-all**. Every physically impaired person has
  a different body, a different injury, and a different daily routine.
- Today, matching a device to a person's *actual* tasks is slow, manual, clinical,
  and expensive — so most people get generic hardware that doesn't fit their life.
- Nobody has an easy way to even **see which daily tasks are hard and why**.

## The solution

**ARMASAI** = wearable perception + AI design, end to end:

1. The user wears **Meta Ray-Ban glasses** and just lives their day.
2. The egocentric video shows ARMASAI **which tasks stand out as difficult, and
   why** (which limb is compensating, what motion fails).
3. ARMASAI **designs a fully capable prosthetic tailored to that routine** —
   sized and articulated for the specific actions that matter to *that* person.
4. It **simulates and validates** the design in physics, then outputs a
   manufacturable CAD model.

The bet: AI reasoning + physics simulation can personalize an assistive limb to a
real life, fast — turning "generic device" into "designed for you."

## How it works (pipeline — Solution / How-it-works slide)

`Ray-Ban clip  →  Gemini perception  →  Design agent  →  MuJoCo simulation  →  CAD model`

- **Perception (Gemini):** reads the clip → the specific action, the affected vs.
  compensating side, range-of-motion, grip needs, and limb measurements.
- **Design agent:** sizes an explicit kinematic arm (links, joints, limits,
  stiffness) from those measurements — different person/routine → different design.
- **Simulation (MuJoCo, in-browser WASM):** runs fixed-seed rollouts to score the
  design (success rate, energy, collisions) before anything is built.
- **CAD:** materializes the validated morphology part-by-part into an exportable
  model.

Live demo runs the whole thing in the browser from a single uploaded clip.

## Team

(Confirm names/handles before presenting — pulled from `docs/WORK_SPLIT.md`.)

- **Benjamin Chan** — Technical Lead: system architecture, AI design reasoning, sim morphology, spatial evaluation, shared contracts *(and the friend whose injury started this)*
- **Vihaan Shringi** — orchestration, APIs, task intake, demo runtime
- **Nathan** — physics environment & scenario fidelity (MuJoCo)
- **Vasi** — policy, RL, reward optimization

## Links / references to access

- GitHub: https://github.com/Benjamin-V-Chan/super-human
- Live demo (Vercel): **TODO — paste the prod URL after `vercel --prod`**
- Repo docs to mine for accurate detail:
  - `docs/PRD.md` — product scope, goals, non-goals
  - `docs/TECHNICAL_PLAN.md` — pipeline contracts, evaluation
  - `docs/WORK_SPLIT.md` — team ownership (source for the Team slide)
  - `docs/BENJI_TASKS_GUIDE.md` — Benji's specific difficult tasks

## Design language (match the product)

- Background `#0a0a0f`, surface `#111118`, border `#2a2a3a`
- Accent cyan `#00d4ff`, accent violet `#7c3aed`, success `#22c55e`, warning `#f59e0b`
- Fonts: **Inter** (headings/body), **JetBrains Mono** (data/labels)
- Mood: human + urgent + hopeful, with engineering credibility. Dark, premium,
  high-contrast. Brand mark: ⬡ ARMASAI.

## Deck spec

- **Exactly 5 slides, short copy** (headline + a few punchy lines each; no walls of text):
  1. **Title / hero** — ⬡ ARMASAI + the one-liner.
  2. **Problem** — Benji's story + the ER photo + "prosthetics aren't one-size-fits-all."
  3. **Solution** — Ray-Ban glasses → personalized prosthetic, the 4 steps.
  4. **How it works** — the 5-stage pipeline, visual, with the in-browser demo callout.
  5. **Team** — the four of us, one line each.
