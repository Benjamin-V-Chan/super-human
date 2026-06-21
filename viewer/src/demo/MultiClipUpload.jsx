import { useCallback, useRef, useState } from 'react'

export default function MultiClipUpload({ clips, onClips, disabled }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  const addFiles = useCallback(async (files) => {
    const valid = Array.from(files).filter((f) =>
      /video\/(mp4|quicktime)/.test(f.type) || /\.(mp4|mov)$/i.test(f.name)
    )
    if (!valid.length) return

    const form = new FormData()
    valid.forEach((f) => form.append('clips', f))
    try {
      const r = await fetch('/api/upload-clips', { method: 'POST', body: form })
      const data = await r.json()
      if (!r.ok) throw new Error(data.error || 'Upload failed')
      const newClips = data.files.map((f, i) => ({
        name: f.name,
        serverPath: f.path,
        sizeMB: (f.size / 1e6).toFixed(1),
        url: URL.createObjectURL(valid[i]),
      }))
      onClips((prev) => {
        const existing = new Set((prev || []).map((c) => c.name))
        return [...(prev || []), ...newClips.filter((c) => !existing.has(c.name))]
      })
    } catch (err) {
      console.error('Upload error:', err.message)
    }
  }, [onClips])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    addFiles(e.dataTransfer.files)
  }, [addFiles])

  const onDragOver = useCallback((e) => { e.preventDefault(); setDragging(true) }, [])
  const onDragLeave = useCallback(() => setDragging(false), [])
  const onInputChange = useCallback((e) => addFiles(e.target.files), [addFiles])
  const removeClip = useCallback((name) => {
    onClips((prev) => (prev || []).filter((c) => c.name !== name))
  }, [onClips])

  return (
    <div className="multi-upload">
      <div
        className={`drop-zone ${dragging ? 'dragging' : ''} ${disabled ? 'disabled' : ''}`}
        onDrop={disabled ? undefined : onDrop}
        onDragOver={disabled ? undefined : onDragOver}
        onDragLeave={disabled ? undefined : onDragLeave}
        onClick={disabled ? undefined : () => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,video/quicktime,.mp4,.mov"
          multiple
          style={{ display: 'none' }}
          onChange={onInputChange}
          disabled={disabled}
        />
        <span className="drop-icon">🎥</span>
        <span className="drop-label">
          {clips?.length ? `${clips.length} clip${clips.length > 1 ? 's' : ''} — drop more` : 'Drop Ray-Ban clips here (or click)'}
        </span>
        <span className="drop-sub">.mp4 / .mov · up to 10 clips · 500 MB each</span>
      </div>

      {clips?.length > 0 && (
        <ul className="clip-list">
          {clips.map((clip, i) => (
            <li key={clip.name} className="clip-item">
              <span className="clip-num">{i + 1}</span>
              {clip.url && (
                <video className="clip-thumb" src={clip.url} muted playsInline preload="metadata" />
              )}
              <div className="clip-meta">
                <span className="clip-name">{clip.name}</span>
                <span className="clip-size">{clip.sizeMB} MB</span>
              </div>
              {!disabled && (
                <button className="clip-remove" onClick={() => removeClip(clip.name)} title="Remove">×</button>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
