import express from 'express'
import cors from 'cors'
import multer from 'multer'
import path from 'path'
import fs from 'fs'
import { spawn } from 'child_process'
import { fileURLToPath } from 'url'
import Anthropic from '@anthropic-ai/sdk'
import { GoogleGenAI } from '@google/genai'
import dotenv from 'dotenv'
import 'dotenv/config'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
// Also load the repo-root .env (GOOGLE_API_KEY / Vertex config live there).
dotenv.config({ path: path.resolve(__dirname, '../.env') })

const app = express()
app.use(cors())
app.use(express.json({ limit: '30mb' })) // base64 video frames

// ── Ray-Ban clip upload → saved into the repo's test_vids/ (ADL clip dir) ──────
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

// ── Gemini perception: analyze Ray-Ban frames → ProblemSpec detection ─────────
function makeGenAI() {
  const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY
  if (apiKey) return new GoogleGenAI({ apiKey })
  if ((process.env.GOOGLE_GENAI_USE_VERTEXAI || '').toLowerCase() === 'true') {
    return new GoogleGenAI({
      vertexai: true,
      project: process.env.GOOGLE_CLOUD_PROJECT,
      location: process.env.GOOGLE_CLOUD_LOCATION || 'us-central1',
    })
  }
  return null
}

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

// frames: array of base64 JPEG strings (no data: prefix)
app.post('/api/analyze-frames', async (req, res) => {
  const { frames = [] } = req.body
  const ai = makeGenAI()
  if (!ai) {
    return res.json({ source: 'unavailable', error: 'No Gemini credentials (set GOOGLE_API_KEY or Vertex ADC)' })
  }
  if (!frames.length) return res.status(400).json({ error: 'No frames provided' })

  try {
    const parts = [
      { text: PERCEPTION_PROMPT },
      // 12 frames is the action-recognition sweet spot (bench: 8 misses the key moment).
      ...frames.slice(0, 12).map((b64) => ({
        inlineData: { mimeType: 'image/jpeg', data: b64.replace(/^data:[^,]+,/, '') },
      })),
    ]
    const result = await ai.models.generateContent({
      model: process.env.GEMMA_MODEL || 'gemini-2.5-flash',
      contents: [{ role: 'user', parts }],
      // temperature=0 for reproducible reads (matches the Python pipeline).
      config: { temperature: 0 },
    })
    const text = result.text ?? result.candidates?.[0]?.content?.parts?.map((p) => p.text).join('') ?? ''
    const detection = extractJson(text)
    if (!detection) return res.json({ source: 'parse_error', raw: text.slice(0, 400) })
    detection.source = 'gemini'
    res.json(detection)
  } catch (err) {
    console.error('analyze-frames:', err.message)
    res.status(502).json({ source: 'error', error: err.message })
  }
})

// ── Real EvalResult: DesignParams → fixed-seed MuJoCo verifier rollouts ──────
const REPO_ROOT = path.resolve(__dirname, '..')
const EVAL_SCRIPT = path.join(REPO_ROOT, 'scripts', 'evaluate_design.py')
const APP_BRIDGE_SCRIPT = path.join(REPO_ROOT, 'scripts', 'app_bridge.py')
const VENV_PYTHON = path.join(REPO_ROOT, '.venv', 'bin', 'python')

function runPythonJson(script, payload, timeoutMs = 30_000, args = []) {
  return new Promise((resolve, reject) => {
    const python = process.env.ARMASAI_PYTHON || (fs.existsSync(VENV_PYTHON) ? VENV_PYTHON : 'python3')
    const child = spawn(python, [script, ...args], { cwd: REPO_ROOT, stdio: ['pipe', 'pipe', 'pipe'] })
    let stdout = '', stderr = ''
    const timer = setTimeout(() => {
      child.kill('SIGTERM')
      reject(new Error('evaluation timed out'))
    }, timeoutMs)
    child.stdout.on('data', (chunk) => { stdout += chunk.toString() })
    child.stderr.on('data', (chunk) => { stderr += chunk.toString() })
    child.on('error', (err) => { clearTimeout(timer); reject(err) })
    child.on('close', (code) => {
      clearTimeout(timer)
      if (code !== 0) return reject(new Error(stderr.trim() || `evaluation exited with code ${code}`))
      try { resolve(JSON.parse(stdout)) }
      catch { reject(new Error('evaluation returned invalid JSON')) }
    })
    child.stdin.end(JSON.stringify(payload))
  })
}

app.post('/api/evaluate-design', async (req, res) => {
  const { problem, design, task_id: taskId } = req.body || {}
  if (!design || typeof design !== 'object') {
    return res.status(400).json({ error: 'DesignParams are required' })
  }
  try {
    const result = await runPythonJson(EVAL_SCRIPT, {
      problem: problem || {},
      design,
      task_id: taskId || 'adl_task_v1',
      seeds: [0, 1, 2],
      n_targets: 2,
      seconds: 1.5,
    })
    res.json(result)
  } catch (err) {
    console.error('evaluate-design:', err.message)
    res.status(502).json({ error: err.message })
  }
})

app.post('/api/analyze-clip', async (req, res) => {
  const requested = String(req.body?.clip_path || '')
  const clipPath = path.resolve(REPO_ROOT, requested)
  if (!requested || !clipPath.startsWith(CLIP_DIR + path.sep) || !fs.existsSync(clipPath)) {
    return res.status(400).json({ error: 'A saved clip under test_vids/ is required' })
  }
  try {
    res.json(await runPythonJson(
      APP_BRIDGE_SCRIPT, { clip_path: clipPath }, 30_000, ['perception'],
    ))
  } catch (err) {
    res.status(502).json({ error: err.message })
  }
})

app.post('/api/derive-design', async (req, res) => {
  try {
    res.json(await runPythonJson(
      APP_BRIDGE_SCRIPT, { problem: req.body?.problem || {} }, 15_000, ['design'],
    ))
  } catch (err) {
    res.status(502).json({ error: err.message })
  }
})

app.post('/api/build-policy', async (req, res) => {
  if (!req.body?.design) return res.status(400).json({ error: 'DesignParams are required' })
  try {
    res.json(await runPythonJson(APP_BRIDGE_SCRIPT, {
      problem: req.body.problem || {}, design: req.body.design, name: req.body.name || 'policy',
    }, 15_000, ['policy']))
  } catch (err) {
    res.status(502).json({ error: err.message })
  }
})

app.post('/api/build-cad', async (req, res) => {
  if (!req.body?.design) return res.status(400).json({ error: 'DesignParams are required' })
  try {
    res.json(await runPythonJson(APP_BRIDGE_SCRIPT, {
      design: req.body.design, name: req.body.name || 'candidate',
    }, 15_000, ['cad']))
  } catch (err) {
    res.status(502).json({ error: err.message })
  }
})

app.post('/api/export-stl', async (req, res) => {
  if (!req.body?.design) return res.status(400).json({ error: 'DesignParams are required' })
  try {
    const artifact = await runPythonJson(APP_BRIDGE_SCRIPT, {
      design: req.body.design, name: req.body.name || 'candidate',
    }, 15_000, ['cad'])
    const artifactPath = path.resolve(REPO_ROOT, artifact.path)
    const stlRoot = path.resolve(REPO_ROOT, 'assets', 'stl')
    if (!artifactPath.startsWith(stlRoot + path.sep)) throw new Error('invalid artifact path')
    res.download(artifactPath, artifact.file)
  } catch (err) {
    res.status(502).json({ error: err.message })
  }
})

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
    if (!process.env.ANTHROPIC_API_KEY) {
      const params = await runPythonJson(APP_BRIDGE_SCRIPT, {
        problem: { primary_action: message || 'general assistive use', tasks: ['reach', 'grasp'] },
      }, 15_000, ['design'])
      sendSSE(res, {
        type: 'done',
        fullText: JSON.stringify({ description: 'Generated by the local Python DesignAgent.', params }),
      })
      return res.end()
    }
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
    if (!process.env.ANTHROPIC_API_KEY) {
      const lower = String(message || '').toLowerCase()
      const updates = {}
      if (lower.includes('titanium')) updates.material = 'titanium'
      if (lower.includes('carbon')) updates.material = 'carbon_fiber'
      const grip = lower.match(/grip(?: width)?\D+(\d+(?:\.\d+)?)\s*(cm|mm|m)?/)
      if (grip) {
        const value = Number(grip[1])
        updates.grip_width = grip[2] === 'mm' ? value / 1000 : grip[2] === 'm' ? value : value / 100
      }
      sendSSE(res, {
        type: 'done',
        fullText: JSON.stringify({
          reply: Object.keys(updates).length
            ? 'Updated the local design parameters.'
            : 'The local design agent is available. Specify a material or grip width to refine the model.',
          params: Object.keys(updates).length ? updates : null,
        }),
      })
      return res.end()
    }
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
