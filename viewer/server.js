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

// ── SSE helpers ───────────────────────────────────────────────────────────────
function sendSSE(res, data) {
  res.write(`data: ${JSON.stringify(data)}\n\n`)
}

function openSSE(res) {
  res.setHeader('Content-Type', 'text/event-stream')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Connection', 'keep-alive')
}

function streamPipelineEvents(child, res, onDone) {
  let buf = ''
  child.stdout.on('data', (chunk) => {
    buf += chunk.toString()
    const lines = buf.split('\n')
    buf = lines.pop()
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue
      try {
        const evt = JSON.parse(trimmed)
        sendSSE(res, evt)
      } catch {
        // non-JSON stdout (print statements etc.) — forward as a log event
        sendSSE(res, { type: 'log', message: trimmed })
      }
    }
  })
  child.stderr.on('data', (chunk) => {
    const msg = chunk.toString().trim()
    if (msg) sendSSE(res, { type: 'log', message: msg })
  })
  child.on('close', (code) => {
    sendSSE(res, { type: 'done', exit_code: code })
    res.end()
    if (onDone) onDone(code)
  })
}

app.post('/api/design', async (req, res) => {
  const { message } = req.body
  openSSE(res)

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
  openSSE(res)

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

// ── Multi-clip upload ─────────────────────────────────────────────────────────
app.post('/api/upload-clips', (req, res) => {
  upload.array('clips', 10)(req, res, (err) => {
    if (err) return res.status(400).json({ error: err.message })
    if (!req.files || req.files.length === 0) return res.status(400).json({ error: 'No files received' })
    res.json({
      saved: true,
      files: req.files.map((f) => ({
        name: f.filename,
        path: `test_vids/${f.filename}`,
        size: f.size,
      })),
    })
  })
})

// ── Scene generation (Nathan's ScenarioAgent + optional Gizmo bake) ───────────
const GEN_SCENE_SCRIPT = path.resolve(REPO_ROOT, 'scripts', 'gen_scene.py')

// In-memory state for the current scene generation job (one at a time).
let _sceneJob = { status: 'idle', stage: null, scenario: null, scene: null, error: null }

app.post('/api/generate-scene', (req, res) => {
  const { action = 'reach for an object', bake_gizmo = false } = req.body || {}
  openSSE(res)
  _sceneJob = { status: 'running', stage: 'scenario', scenario: null, scene: null, error: null }

  const pythonBin = fs.existsSync(path.join(REPO_ROOT, '.venv', 'bin', 'python3'))
    ? path.join(REPO_ROOT, '.venv', 'bin', 'python3')
    : (process.env.ARMASAI_PYTHON || 'python3')

  const child = spawn(pythonBin, [GEN_SCENE_SCRIPT], {
    env: { ...process.env, PYTHONPATH: REPO_ROOT },
    timeout: 10 * 60 * 1000, // 10 min (Gizmo bake can take a few minutes)
  })
  child.stdin.write(JSON.stringify({ action, bake_gizmo }))
  child.stdin.end()

  streamPipelineEvents(child, res, (code) => {
    _sceneJob.status = code === 0 ? 'done' : 'error'
    if (code !== 0) _sceneJob.error = `exit code ${code}`
  })

  // Also parse events to update in-memory state for polling
  child.stdout.on('data', (chunk) => {
    for (const line of chunk.toString().split('\n')) {
      const t = line.trim()
      if (!t) continue
      try {
        const evt = JSON.parse(t)
        if (evt.type === 'status') _sceneJob.stage = evt.stage_name || evt.stage
        if (evt.type === 'done') { _sceneJob.scenario = evt.scenario; _sceneJob.scene = evt.scene }
        if (evt.type === 'error') _sceneJob.error = evt.message
      } catch { /* ignore */ }
    }
  })
})

app.get('/api/scene-status', (_req, res) => {
  res.json(_sceneJob)
})

app.get('/api/scene-showcase', (req, res) => {
  // Return the showcase scenario instantly (ADL library, no bake needed). The
  // gen_scene.py script emits multiple NDJSON lines; we grab the last "done" one.
  const action = req.query.action || 'reach for an object'
  const pythonBin = fs.existsSync(path.join(REPO_ROOT, '.venv', 'bin', 'python3'))
    ? path.join(REPO_ROOT, '.venv', 'bin', 'python3')
    : (process.env.ARMASAI_PYTHON || 'python3')

  const child = spawn(pythonBin, [GEN_SCENE_SCRIPT], {
    env: { ...process.env, PYTHONPATH: REPO_ROOT },
    timeout: 15_000,
  })
  child.stdin.write(JSON.stringify({ action, bake_gizmo: false }))
  child.stdin.end()

  let buf = ''
  child.stdout.on('data', (chunk) => { buf += chunk.toString() })
  child.on('close', () => {
    const lines = buf.split('\n').filter((l) => l.trim())
    let last = null
    for (const line of lines) {
      try { const evt = JSON.parse(line); if (evt.type === 'done') last = evt } catch { /* skip */ }
    }
    if (last) res.json(last)
    else res.status(500).json({ error: 'scene generation returned no result' })
  })
})

// ── Multi-clip pipeline (SSE stream) ──────────────────────────────────────────
// Spawns Python run_multi_pipeline.py and forwards newline-delimited JSON events
// as SSE to the browser.
const MULTI_PIPELINE_SCRIPT = path.resolve(REPO_ROOT, 'scripts', 'run_multi_pipeline.py')

app.post('/api/run-multi-pipeline', (req, res) => {
  const { clip_paths: clipPaths = [], quick_mode: quickMode = false } = req.body || {}
  if (!Array.isArray(clipPaths) || clipPaths.length === 0) {
    return res.status(400).json({ error: 'clip_paths array is required' })
  }

  // Validate paths are under test_vids/
  const resolvedPaths = []
  for (const p of clipPaths) {
    const full = path.resolve(REPO_ROOT, p)
    if (!full.startsWith(CLIP_DIR + path.sep) || !fs.existsSync(full)) {
      return res.status(400).json({ error: `Invalid or missing clip path: ${p}` })
    }
    resolvedPaths.push(full)
  }

  openSSE(res)
  sendSSE(res, { type: 'pipeline_start', clip_paths: clipPaths, n_clips: clipPaths.length })

  const pythonBin = process.env.ARMASAI_PYTHON
    || path.join(REPO_ROOT, '.venv', 'bin', 'python3')
    || 'python3'

  const child = spawn(
    fs.existsSync(path.join(REPO_ROOT, '.venv', 'bin', 'python3'))
      ? path.join(REPO_ROOT, '.venv', 'bin', 'python3')
      : (process.env.ARMASAI_PYTHON || 'python3'),
    [MULTI_PIPELINE_SCRIPT],
    {
      env: { ...process.env, PYTHONPATH: REPO_ROOT },
      timeout: 30 * 60 * 1000, // 30 min max
    }
  )

  child.stdin.write(JSON.stringify({ clip_paths: resolvedPaths, quick_mode: quickMode }))
  child.stdin.end()

  streamPipelineEvents(child, res, null)
})

// ── Clip listing and thumbnails ───────────────────────────────────────────────
app.get('/api/clips', (_req, res) => {
  try {
    const files = fs.readdirSync(CLIP_DIR).filter((f) => /\.(mp4|mov)$/i.test(f))
    res.json(files.map((f) => ({ name: f, path: `test_vids/${f}` })))
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

app.get('/api/thumbnail', (req, res) => {
  const clip = String(req.query.clip || '')
  const full = path.resolve(REPO_ROOT, clip)
  if (!clip || !full.startsWith(CLIP_DIR + path.sep) || !fs.existsSync(full)) {
    return res.status(404).send('Not found')
  }
  res.setHeader('Content-Type', 'image/jpeg')
  const child = spawn('ffmpeg', [
    '-i', full, '-frames:v', '1', '-q:v', '5', '-f', 'image2', '-vcodec', 'mjpeg', 'pipe:1',
  ], { stdio: ['ignore', 'pipe', 'ignore'] })
  child.stdout.pipe(res)
  child.on('error', () => {
    if (!res.headersSent) {
      // 1×1 transparent GIF placeholder when ffmpeg unavailable
      res.setHeader('Content-Type', 'image/gif')
      res.end(Buffer.from('R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7', 'base64'))
    }
  })
})

// ── Single-clip pipeline entry point (called by pipeline.html dashboard) ──────
// Accepts { clip, quick } and delegates to run_multi_pipeline.py
app.post('/api/run', (req, res) => {
  const { clip, quick = false } = req.body || {}
  if (!clip) return res.status(400).json({ error: 'clip is required' })
  const full = path.resolve(REPO_ROOT, clip)
  if (!full.startsWith(CLIP_DIR + path.sep) || !fs.existsSync(full)) {
    return res.status(400).json({ error: `Invalid or missing clip path: ${clip}` })
  }

  openSSE(res)
  sendSSE(res, { type: 'pipeline_start', clip_paths: [clip], n_clips: 1 })

  const pythonBin = fs.existsSync(path.join(REPO_ROOT, '.venv', 'bin', 'python3'))
    ? path.join(REPO_ROOT, '.venv', 'bin', 'python3')
    : (process.env.ARMASAI_PYTHON || 'python3')

  const child = spawn(pythonBin, [MULTI_PIPELINE_SCRIPT], {
    env: { ...process.env, PYTHONPATH: REPO_ROOT },
    timeout: 30 * 60 * 1000,
  })
  child.stdin.write(JSON.stringify({ clip_paths: [full], quick_mode: !!quick }))
  child.stdin.end()

  streamPipelineEvents(child, res, null)
})

// ── MJCF scene serving ────────────────────────────────────────────────────────
const MJCF_DIR = path.resolve(REPO_ROOT, 'assets', 'mjcf')

app.get('/api/scene', (req, res) => {
  const name = String(req.query.name || 'default').replace(/[^a-zA-Z0-9_\-]/g, '')
  let xmlPath = path.resolve(MJCF_DIR, `${name}.xml`)
  if (!fs.existsSync(xmlPath)) {
    xmlPath = path.resolve(MJCF_DIR, 'default.xml')
    if (!fs.existsSync(xmlPath)) return res.status(404).send('Scene not found')
  }
  res.setHeader('Content-Type', 'application/xml')
  res.sendFile(xmlPath)
})

// ── STL mesh serving ──────────────────────────────────────────────────────────
const STL_ROOT = path.resolve(REPO_ROOT, 'assets', 'stl')

app.get('/api/mesh', (req, res) => {
  const name = String(req.query.name || 'default').replace(/[^a-zA-Z0-9_\-]/g, '')
  const link = String(req.query.link || '').replace(/[^a-zA-Z0-9_\-]/g, '')
  if (!link) return res.status(400).send('link param required')
  // Try scene-specific dir first, then default dir
  for (const dir of [name, 'default']) {
    const stlPath = path.resolve(STL_ROOT, dir, `${link}.stl`)
    if (stlPath.startsWith(STL_ROOT) && fs.existsSync(stlPath)) {
      return res.sendFile(stlPath)
    }
  }
  res.status(404).send('Mesh not found')
})

// ── Static webdemo (pipeline dashboard, MuJoCo demos) ────────────────────────
app.use(express.static(path.resolve(__dirname, '../webdemo')))

// ── App listen ────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3001
app.listen(PORT, () => console.log(`Armasai server :${PORT}`))
