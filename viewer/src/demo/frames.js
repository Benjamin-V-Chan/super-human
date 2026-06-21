// Extract N evenly-spaced JPEG frames from a video URL, client-side (no ffmpeg).
// Returns dataURL strings. Resolves to [] if the browser can't decode the clip
// (e.g. some .mov codecs in Chrome) so callers can fall back gracefully.
export async function extractFrames(url, count = 6, maxW = 512) {
  return new Promise((resolve) => {
    const v = document.createElement('video')
    v.muted = true
    v.crossOrigin = 'anonymous'
    v.preload = 'auto'
    v.src = url

    const frames = []
    let times = []
    let i = 0
    let settled = false
    const done = () => { if (!settled) { settled = true; resolve(frames) } }

    // bail out if the clip never decodes
    const bail = setTimeout(done, 12000)
    v.onerror = () => { clearTimeout(bail); done() }

    v.onloadeddata = () => {
      const dur = isFinite(v.duration) && v.duration > 0 ? v.duration : 0
      if (!dur) { clearTimeout(bail); return done() }
      // sample within [5%, 95%] of the clip
      times = Array.from({ length: count }, (_, k) =>
        dur * (0.05 + (0.9 * k) / Math.max(1, count - 1)))
      seekNext()
    }

    const canvas = document.createElement('canvas')
    const seekNext = () => {
      if (i >= times.length) { clearTimeout(bail); return done() }
      v.currentTime = times[i]
    }
    v.onseeked = () => {
      try {
        const scale = Math.min(1, maxW / (v.videoWidth || maxW))
        canvas.width = Math.round((v.videoWidth || maxW) * scale)
        canvas.height = Math.round((v.videoHeight || maxW * 0.56) * scale)
        const ctx = canvas.getContext('2d')
        ctx.drawImage(v, 0, 0, canvas.width, canvas.height)
        frames.push(canvas.toDataURL('image/jpeg', 0.7))
      } catch { /* tainted/undecodable frame — skip */ }
      i += 1
      seekNext()
    }
  })
}
