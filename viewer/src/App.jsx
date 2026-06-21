import { useState, useCallback } from 'react'
import Viewer3D from './components/Viewer3D.jsx'
import ChatPanel from './components/ChatPanel.jsx'
import ParameterPanel from './components/ParameterPanel.jsx'
import { DEFAULT_PARAMS } from './lib/defaults.js'
import { downloadStl, streamDesign, streamChat } from './lib/api.js'
import './App.css'

export default function App() {
  const [params, setParams] = useState(DEFAULT_PARAMS)
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Describe the patient — residual limb length, activity level, grip requirements — and I\'ll generate a personalized prosthetic arm design.',
    },
  ])
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('chat') // 'chat' | 'params'
  const [wireframe, setWireframe] = useState(false)
  const [showDimensions, setShowDimensions] = useState(true)
  const [exporting, setExporting] = useState(false)

  const mergeParams = useCallback((updates) => {
    setParams((prev) => ({ ...prev, ...updates }))
  }, [])

  const handleSend = useCallback(async (text) => {
    const userMsg = { role: 'user', content: text }
    setMessages((m) => [...m, userMsg])
    setLoading(true)

    const history = messages
      .filter((m) => m.role !== 'system')
      .map((m) => ({ role: m.role, content: m.content }))

    // First message or explicit "design" request → use /api/design to generate fresh params
    const isInitialDesign = messages.length === 1 ||
      /design|generat|creat|build|make|new|start/i.test(text)

    let reply = ''
    let assistantMsg = { role: 'assistant', content: '' }
    setMessages((m) => [...m, assistantMsg])

    if (isInitialDesign) {
      let fullText = ''
      await streamDesign(text, (chunk) => {
        fullText += chunk
        setMessages((m) => {
          const updated = [...m]
          updated[updated.length - 1] = { role: 'assistant', content: '⚙ Generating design...' }
          return updated
        })
      })

      try {
        const json = JSON.parse(fullText.trim())
        if (json.params) {
          mergeParams(json.params)
          reply = `✓ ${json.description}`
        } else {
          reply = 'Could not parse design params. Try describing the patient more specifically.'
        }
      } catch {
        reply = 'Could not parse response. Please try again.'
      }
    } else {
      let fullText = ''
      await streamChat(text, history, params, (chunk) => {
        fullText += chunk
        setMessages((m) => {
          const updated = [...m]
          updated[updated.length - 1] = { role: 'assistant', content: fullText }
          return updated
        })
      })

      try {
        const json = JSON.parse(fullText.trim())
        reply = json.reply || fullText
        if (json.params) mergeParams(json.params)
      } catch {
        reply = fullText
      }
    }

    setMessages((m) => {
      const updated = [...m]
      updated[updated.length - 1] = { role: 'assistant', content: reply }
      return updated
    })
    setLoading(false)
  }, [messages, params, mergeParams])

  const handleReset = () => {
    setParams(DEFAULT_PARAMS)
    setMessages([{
      role: 'assistant',
      content: 'Design reset. Describe a new patient to generate a custom arm.',
    }])
  }

  const handleExport = async (event) => {
    event.preventDefault()
    if (exporting) return
    setExporting(true)
    try { await downloadStl(params, 'studio_candidate') }
    catch (error) {
      setMessages((m) => [...m, { role: 'assistant', content: `Export failed: ${error.message}` }])
    } finally { setExporting(false) }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-brand">
          <span className="logo-mark">⬡</span>
          <span className="logo-text">ARMASAI</span>
          <span className="logo-sub">Prosthetic CAD</span>
        </div>
        <div className="header-controls">
          <a href="#" className="export-btn" style={{ marginRight: 8 }}
            onClick={(e) => { e.preventDefault(); window.location.hash = '' }}>
            ← Pipeline Demo
          </a>
          <button
            className={`icon-btn ${showDimensions ? 'active' : ''}`}
            onClick={() => setShowDimensions((v) => !v)}
            title="Toggle dimensions"
          >
            ⊕
          </button>
          <button
            className={`icon-btn ${wireframe ? 'active' : ''}`}
            onClick={() => setWireframe((v) => !v)}
            title="Wireframe"
          >
            ⊞
          </button>
          <button className="icon-btn" onClick={handleReset} title="Reset design">
            ↺
          </button>
          <div className="header-divider" />
          <a
            href="#"
            className="export-btn"
            onClick={handleExport}
          >
            {exporting ? 'Exporting…' : 'Export STL'}
          </a>
        </div>
      </header>

      <div className="main">
        <aside className="sidebar">
          <div className="tab-bar">
            <button
              className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              AI Design
            </button>
            <button
              className={`tab ${activeTab === 'params' ? 'active' : ''}`}
              onClick={() => setActiveTab('params')}
            >
              Parameters
            </button>
          </div>

          {activeTab === 'chat' ? (
            <ChatPanel
              messages={messages}
              onSend={handleSend}
              loading={loading}
            />
          ) : (
            <ParameterPanel params={params} onChange={mergeParams} />
          )}
        </aside>

        <main className="viewer-wrap">
          <Viewer3D
            params={params}
            wireframe={wireframe}
            showDimensions={showDimensions}
          />
        </main>
      </div>
    </div>
  )
}
