import { useEffect, useState } from 'react'

// Minimal syntax-highlighted, optionally typewritered JSON block.
export default function JsonBlock({ data, animate = false, speed = 8 }) {
  const full = JSON.stringify(data, null, 2)
  const [shown, setShown] = useState(animate ? '' : full)

  useEffect(() => {
    if (!animate) { setShown(full); return }
    setShown('')
    let i = 0
    const step = Math.max(1, Math.round(full.length / 60))
    const id = setInterval(() => {
      i += step
      setShown(full.slice(0, i))
      if (i >= full.length) clearInterval(id)
    }, speed)
    return () => clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [full, animate])

  return (
    <pre className="json-block" dangerouslySetInnerHTML={{ __html: colorize(shown) }} />
  )
}

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function colorize(src) {
  const esc = escapeHtml(src)
  return esc.replace(
    /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(\.\d+)?([eE][+-]?\d+)?)/g,
    (m) => {
      let cls = 'tok-num'
      if (/^"/.test(m)) cls = /:$/.test(m) ? 'tok-key' : 'tok-str'
      else if (/true|false/.test(m)) cls = 'tok-bool'
      else if (/null/.test(m)) cls = 'tok-null'
      return `<span class="${cls}">${m}</span>`
    },
  )
}
