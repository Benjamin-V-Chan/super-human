// Vercel serverless function: Gemini perception over Ray-Ban frames.
// API-key path only (GOOGLE_API_KEY / GEMINI_API_KEY) — never Vertex ADC, which
// has no credentials on Vercel. Standalone (does not import server.js).
import { GoogleGenAI } from '@google/genai'

const PERCEPTION_PROMPT = `You are the perception module of a custom upper-limb prosthetic design system.
Every subject is a candidate for an upper-limb prosthesis: one arm/hand is absent or non-functional,
the other compensates. Read the functional situation from these sequential video frames.

- The hand that does the work (reaching, holding, manipulating) is the RESIDUAL (functioning) side.
- The opposite side is AFFECTED and needs the prosthesis. Decide sides ONLY from the footage.
- Name the SPECIFIC action: concrete object + precise verb (e.g. "unscrewing a bottle cap",
  "tearing a sheet of paper", "pouring water into a cup"). Do not default to any action.

Reason through the action internally before answering:
- What object is in the working hand, and what is the hand DOING to it
  (twisting, tearing, folding, lifting, pouring, cutting)?
- With ONE hand, the person BRACES objects against a surface (sill, table, lap, body) to
  replace the missing hand. A hand pressing paper onto a sill is almost certainly TEARING or
  FOLDING it, NOT wiping the surface. Report the intended task, not the incidental contact.
- Estimate residual_anthropometrics (the intact arm, in meters) so the prosthesis can be sized
  to MIRROR the contralateral limb; use the visible arm for scale or assume adult proportions.

Respond with ONLY a JSON object of this exact shape:
{
  "primary_action": "<specific object + verb>",
  "affected_side": "<left|right>",
  "residual_side": "<left|right>",
  "tasks": ["reach","grasp"],
  "rom": {"shoulder_flexion": 110, "elbow_flexion": 130, "wrist_rotation": 60},
  "residual_strength": {"shoulder": 0.7, "elbow": 0.6},
  "grip_capacity": 0.4,
  "residual_anthropometrics": {"upper_arm_len": 0.30, "forearm_len": 0.26, "hand_length": 0.19, "grip_span": 0.08},
  "pain_points": ["..."]
}`

function extractJson(text) {
  const m = text && text.match(/\{[\s\S]*\}/)
  if (!m) return null
  try { return JSON.parse(m[0]) } catch { return null }
}

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' })

  const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY
  if (!apiKey) return res.status(200).json({ source: 'unavailable', error: 'GOOGLE_API_KEY not set' })

  const frames = req.body?.frames || []
  if (!frames.length) return res.status(400).json({ error: 'No frames provided' })

  try {
    const ai = new GoogleGenAI({ apiKey })
    const parts = [
      { text: PERCEPTION_PROMPT },
      ...frames.slice(0, 12).map((b64) => ({
        inlineData: { mimeType: 'image/jpeg', data: String(b64).replace(/^data:[^,]+,/, '') },
      })),
    ]
    const result = await ai.models.generateContent({
      model: process.env.GEMINI_MODEL || process.env.GEMMA_MODEL || 'gemini-2.5-flash',
      contents: [{ role: 'user', parts }],
      config: { temperature: 0 },
    })
    const text = result.text ?? result.candidates?.[0]?.content?.parts?.map((p) => p.text).join('') ?? ''
    const detection = extractJson(text)
    if (!detection) return res.status(200).json({ source: 'parse_error', raw: text.slice(0, 400) })
    detection.source = 'gemini'
    return res.status(200).json(detection)
  } catch (err) {
    const quota = /quota|rate|429/i.test(err?.message || '')
    return res.status(200).json({ source: quota ? 'rate_limited' : 'error', error: err?.message })
  }
}
