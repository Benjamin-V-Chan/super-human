import { useState } from 'react'

// Modal that lets an operator "complete" an unfinished integration by pasting
// the real output payload. Until then the pipeline runs on the expected shape.
export default function IntegrationGate({ stage, current, onClose, onSave }) {
  const template = JSON.stringify(current || stage.expected, null, 2)
  const [text, setText] = useState(template)
  const [error, setError] = useState('')

  const save = () => {
    try {
      const parsed = JSON.parse(text)
      setError('')
      onSave(parsed)
    } catch (e) {
      setError('Invalid JSON — ' + e.message)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="modal-title">{stage.icon} Complete · {stage.name}</div>
            <div className="modal-sub">
              {stage.consumes} <span className="arrow">→</span> {stage.emits}
              {stage.owner && <> · owned by <b>{stage.owner}</b></>}
            </div>
          </div>
          <button className="modal-x" onClick={onClose}>✕</button>
        </div>

        <p className="modal-note">
          This stage isn’t wired yet. Paste the real <b>{stage.emits}</b> from your
          integration, or keep the expected output below as a placeholder.
        </p>

        <textarea
          className="modal-input"
          value={text}
          spellCheck={false}
          onChange={(e) => setText(e.target.value)}
          rows={14}
        />
        {error && <div className="modal-err">{error}</div>}

        <div className="modal-actions">
          <button className="btn ghost" onClick={() => setText(JSON.stringify(stage.expected, null, 2))}>
            Reset to expected
          </button>
          <div className="spacer" />
          <button className="btn ghost" onClick={onClose}>Cancel</button>
          <button className="btn primary" onClick={save}>Save output</button>
        </div>
      </div>
    </div>
  )
}
