import express from 'express'
import cors from 'cors'
import multer from 'multer'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'
import Anthropic from '@anthropic-ai/sdk'
import 'dotenv/config'

const app = express()
app.use(cors())
app.use(express.json())

// ── Ray-Ban clip upload → saved into the repo's test_vids/ (ADL clip dir) ──────
const __dirname = path.dirname(fileURLToPath(import.meta.url))
const CLIP_DIR = path.resolve(__dirname, '../test_vids')
fs.mkdirSync(CLIP_DIR, { recursive: true })

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, CLIP_DIR),
  filename: (_req, file, cb) => cb(null, file.originalname.replace(/[^\w.\- ]+/g, '_')),
})
const upload = multer({
  storage,
  limits: { fileSize: 500 * 1024 * 1024 }, // 500 MB
  fileFilter: (_req, file, cb) => {
    const ok = /video\/(mp4|quicktime)/.test(file.mimetype) || /\.(mp4|mov)$/i.test(file.originalname)
    cb(ok ? null : new Error('Only .mp4 / .mov files are accepted'), ok)
  },
})

app.post('/api/upload-clip', (req, res) => {
  upload.single('clip')(req, res, (err) => {
    if (err) return res.status(400).json({ error: err.message })
    if (!req.file) return res.status(400).json({ error: 'No file received' })
    res.json({
      saved: true,
      name: req.file.filename,
      path: `test_vids/${req.file.filename}`,
      size: req.file.size,
    })
  })
})

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })

const DESIGN_SYSTEM = `You are an expert prosthetist and biomedical engineer specializing in upper-limb prosthetics.

Given a natural language description of a patient's needs, produce precise DesignParams for a custom prosthetic arm.

Respond ONLY with a valid JSON object — no markdown, no explanation, just JSON:
{
  "description": "<1-sentence summary of the design>",
  "params": {
    "upper_arm_len": <meters, 0.20-0.40, default 0.30>,
    "forearm_len": <meters, 0.18-0.32, default 0.26>,
    "joint_stiffness": <N·m/rad, 0.5-3.0, default 1.0>,
    "grip_width": <meters, 0.04-0.14, default 0.08>,
    "joint_limits": {
      "elbow": [<min_degrees, -10 to 0>, <max_degrees, 90-150>],
      "wrist": [<min_degrees, -60 to -30>, <max_degrees, 30-60>]
    },
    "elbow_angle": <current rest angle in degrees, 0=straight, 90=right-angle, default 30>,
    "arm_radius": <tube radius meters, 0.020-0.045, default 0.030>,
    "primaryColor": "<hex, e.g. #0d1117 for carbon, #c0c0c0 for titanium>",
    "accentColor": "<hex, accent/joint color>",
    "material": "carbon_fiber" | "titanium" | "polymer"
  }
}

Infer from context:
- Active/athletic users → lower stiffness (0.8), wider grip, longer segments
- Older/weaker users → higher stiffness (1.5+), narrower grip
- Child → scale all lengths down 20-30%
- Heavy-duty work → titanium + wider grip, higher stiffness
- Aesthetic preference → carbon_fiber with neon accents
- Missing info → use safe defaults for average adult`

const CHAT_SYSTEM = (params) => `You are an expert prosthetist helping refine a prosthetic arm design in the Armasai CAD system.

Current design parameters:
${JSON.stringify(params, null, 2)}

When the user wants changes, respond with JSON:
{"reply": "<conversational response>", "params": {<only changed params>}}

When answering questions only:
{"reply": "<answer>", "params": null}

Be brief and precise. You can explain biomechanical trade-offs.`

// SSE helper
function sendSSE(res, data) {
  res.write(`data: ${JSON.stringify(data)}\n\n`)
}

app.post('/api/design', async (req, res) => {
  const { message } = req.body
  res.setHeader('Content-Type', 'text/event-stream')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Connection', 'keep-alive')

  try {
    const stream = await client.messages.stream({
      model: 'claude-sonnet-4-6',
      max_tokens: 800,
      system: DESIGN_SYSTEM,
      messages: [{ role: 'user', content: message }],
    })

    let fullText = ''
    for await (const chunk of stream) {
      if (chunk.type === 'content_block_delta' && chunk.delta.type === 'text_delta') {
        fullText += chunk.delta.text
        sendSSE(res, { type: 'delta', text: chunk.delta.text })
      }
    }
    sendSSE(res, { type: 'done', fullText })
    res.end()
  } catch (err) {
    console.error(err)
    sendSSE(res, { type: 'error', message: err.message })
    res.end()
  }
})

app.post('/api/chat', async (req, res) => {
  const { message, history = [], params } = req.body
  res.setHeader('Content-Type', 'text/event-stream')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Connection', 'keep-alive')

  try {
    const stream = await client.messages.stream({
      model: 'claude-sonnet-4-6',
      max_tokens: 512,
      system: CHAT_SYSTEM(params),
      messages: [...history, { role: 'user', content: message }],
    })

    let fullText = ''
    for await (const chunk of stream) {
      if (chunk.type === 'content_block_delta' && chunk.delta.type === 'text_delta') {
        fullText += chunk.delta.text
        sendSSE(res, { type: 'delta', text: chunk.delta.text })
      }
    }
    sendSSE(res, { type: 'done', fullText })
    res.end()
  } catch (err) {
    console.error(err)
    sendSSE(res, { type: 'error', message: err.message })
    res.end()
  }
})

const PORT = process.env.PORT || 3001
app.listen(PORT, () => console.log(`Armasai server :${PORT}`))
