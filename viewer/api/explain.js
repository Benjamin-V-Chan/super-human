// Vercel serverless function: one quick Claude Haiku call that turns a pipeline
// stage's data into a plain-English sentence. Returns 503 if no key (the client
// then falls back to its deterministic explanation). Standalone — no server.js.
import Anthropic from '@anthropic-ai/sdk'

const STAGE_HINT = {
  capture: 'the input clip',
  perception: 'what the vision model saw and which arm needs the prosthesis',
  design: 'the prosthetic arm it sized',
  simulation: 'how the design did in physics simulation',
  policy: 'the controller it built',
  cad: 'the assembled printable model',
}

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' })
  if (!process.env.ANTHROPIC_API_KEY) return res.status(503).json({ error: 'no anthropic key' })

  const { stage, data } = req.body || {}
  try {
    const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })
    const msg = await client.messages.create({
      model: 'claude-haiku-4-5',
      max_tokens: 90,
      messages: [{
        role: 'user',
        content:
          `You explain one step of an AI pipeline that designs custom prosthetic arms from a video. ` +
          `In ONE or TWO plain sentences a non-expert can understand, say what this "${stage}" step ` +
          `(${STAGE_HINT[stage] || stage}) concluded. No jargon, no JSON, no field names, no markdown. ` +
          `Data:\n${JSON.stringify(data).slice(0, 1200)}`,
      }],
    })
    const text = (msg.content?.[0]?.text || '').trim()
    if (!text) return res.status(502).json({ error: 'empty' })
    return res.status(200).json({ text })
  } catch (err) {
    return res.status(502).json({ error: err?.message })
  }
}
