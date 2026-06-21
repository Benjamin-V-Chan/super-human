import { useRef, useState } from 'react'
import RayBanPOV from './RayBanPOV.jsx'

// Controlled Ray-Ban clip uploader. Accepts .mp4 / .mov via picker or drag-drop,
// previews it inline (HTML5 video with the capture HUD), and POSTs it to
// /api/upload-clip so it lands in the repo's test_vids/ ADL dir. Falls back to
// the faux POV when no clip is loaded; preview still works if the server is down.
export default function RayBanUpload({ clip, onClip, sampling = false }) {
  const inputRef = useRef(null)
  const [drag, setDrag] = useState(false)
  const [busy, setBusy] = useState(false)

  const pick = () => inputRef.current?.click()

  const accept = (file) => {
    if (!file) return
    const okType = /\.(mp4|mov)$/i.test(file.name) || /video\/(mp4|quicktime)/.test(file.type)
    if (!okType) {
      onClip({ error: 'Only .mp4 / .mov files are accepted' })
      return
    }
    const url = URL.createObjectURL(file)
    // probe duration off-screen
    const probe = document.createElement('video')
    probe.preload = 'metadata'
    probe.onloadedmetadata = () => {
      onClip({
        url,
        name: file.name,
        sizeMB: (file.size / 1048576).toFixed(1),
        durationS: isFinite(probe.duration) ? probe.duration.toFixed(1) : null,
        saved: false,
      })
    }
    probe.onerror = () => onClip({ url, name: file.name, sizeMB: (file.size / 1048576).toFixed(1), durationS: null, saved: false })
    probe.src = url

    // best-effort server save → test_vids/
    setBusy(true)
    const form = new FormData()
    form.append('clip', file)
    fetch('/api/upload-clip', { method: 'POST', body: form })
      .then((r) => r.json())
      .then((d) => {
        setBusy(false)
        if (d?.saved) onClip((c) => ({ ...(c || {}), saved: true, serverPath: d.path }))
      })
      .catch(() => setBusy(false)) // preview still works without backend
  }

  const onDrop = (e) => {
    e.preventDefault(); setDrag(false)
    accept(e.dataTransfer.files?.[0])
  }

  return (
    <div className="rb-upload">
      <input
        ref={inputRef} type="file" accept="video/mp4,video/quicktime,.mp4,.mov"
        style={{ display: 'none' }}
        onChange={(e) => accept(e.target.files?.[0])}
      />

      {clip?.url ? (
        <div className={`rb-preview ${sampling ? 'sampling' : ''}`}>
          <video src={clip.url} controls muted loop playsInline className="rb-video" />
          <div className="pov-hud top" style={{ pointerEvents: 'none' }}>
            <span className="pov-rec on">● REC</span>
            <span className="pov-dev">RAY-BAN META</span>
          </div>
          {sampling && (
            <>
              <span className="pov-bracket tl" /><span className="pov-bracket tr" />
              <span className="pov-bracket bl" /><span className="pov-bracket br" />
            </>
          )}
        </div>
      ) : (
        <div
          className={`rb-drop ${drag ? 'over' : ''}`}
          onClick={pick}
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
        >
          <div className="rb-fakepov"><RayBanPOV recording sampling={false} taskHint="sample clip" /></div>
          <div className="rb-drop-cta">
            <span className="rb-up-icon">⬆</span>
            <b>Upload Ray-Ban clip</b>
            <span className="rb-up-sub">drag &amp; drop or click · .mp4 / .mov</span>
          </div>
        </div>
      )}

      <div className="rb-meta">
        {clip?.error && <span className="rb-err">⚠ {clip.error}</span>}
        {clip?.url && !clip.error && (
          <>
            <span className="rb-name" title={clip.name}>{clip.name}</span>
            <span className="rb-stat">
              {clip.sizeMB} MB{clip.durationS ? ` · ${clip.durationS}s` : ''}
              {busy ? ' · saving…' : clip.saved ? ' · ✓ test_vids/' : ''}
            </span>
            <button className="btn tiny" onClick={(e) => { e.stopPropagation(); pick() }}>Replace clip</button>
          </>
        )}
      </div>
    </div>
  )
}
