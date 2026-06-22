---
description: Build the Super Human 5-slide pitch deck (problem / solution / team) from the saved pitch context + Benji ER photo
argument-hint: "[optional: audience or tweak, e.g. 'investors, 90s read']"
---

# /pitch-deck — Super Human pitch deck

You are co-building a **5-slide pitch deck** for **Super Human**. Work from the saved
context — do NOT invent facts.

## Sources to read first (in this order)

1. `docs/pitch/CONTEXT.md` — **primary source** (story, problem, solution, team,
   links, design tokens, deck spec). Follow it.
2. For accurate technical/team detail, also skim:
   - `docs/PRD.md`, `docs/TECHNICAL_PLAN.md` (pipeline + scope)
   - `docs/WORK_SPLIT.md` (the real team + roles — use for the Team slide)
   - `docs/BENJI_TASKS_GUIDE.md` (Benji's specific hard tasks — good Problem detail)
3. If `CONTEXT.md` has a live Vercel URL filled in, include it on the Title slide;
   otherwise omit it (don't fabricate one).

## Image

Use `assets/pitch/benjihospital.jpg` (Benji in the ER) on the **Problem** slide.
It's a real photo of our friend — present it with respect, not shock value.

## Build

Produce **one self-contained HTML deck** at `docs/pitch/superhuman-deck.html` using
**reveal.js via CDN** (so arrow keys / space navigate, 16:9). Embed the photo by
relative path (`../../assets/pitch/benjihospital.jpg`). No build step required —
it must open directly in a browser.

**Exactly 5 slides, short + punchy** (headline + a few lines; never a wall of text):

1. **Title / hero** — `⬡ Super Human` + the one-liner. Subtle cyan/violet glow.
2. **Problem** — Benji's story (workout → torn muscle → rhabdomyolysis → ER →
   everyday tasks now hard) with the ER photo, landing on *"prosthetics aren't
   one-size-fits-all."*
3. **Solution** — Meta Ray-Ban glasses → see which tasks are hard & why → design a
   prosthetic tailored to that person's routine. Show the 4 steps.
4. **How it works** — the pipeline `Ray-Ban clip → Gemini → Design → MuJoCo → CAD`
   as a clean left-to-right visual; call out that the demo runs in-browser.
5. **Team** — the four teammates from `WORK_SPLIT.md`, one line each (note Benji is
   both a teammate and the friend who inspired it).

## Style (match the product — tokens are in CONTEXT.md)

Dark premium: bg `#0a0a0f`, surface `#111118`, accent cyan `#00d4ff`, violet
`#7c3aed`, success `#22c55e`. Fonts **Inter** (headings/body) + **JetBrains Mono**
(labels/data) via Google Fonts CDN. High contrast, generous whitespace, ⬡ brand
mark. Human + urgent + hopeful, with engineering credibility.

## Finish

- Save to `docs/pitch/superhuman-deck.html`, then **open it** (`open docs/pitch/superhuman-deck.html`)
  so the user can present immediately.
- Print a one-line summary of each slide and remind the user to paste the live
  Vercel URL into `docs/pitch/CONTEXT.md` once deployed, then re-run `/pitch-deck`.

$ARGUMENTS
