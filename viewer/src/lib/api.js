async function consumeSSE(url, body, onDelta) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) throw new Error(`HTTP ${res.status}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let fullText = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split('\n')
    buffer = lines.pop() // keep incomplete line

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      let event
      try { event = JSON.parse(line.slice(6)) } catch { continue }
      if (event.type === 'delta') {
        onDelta(event.text)
        fullText += event.text
      } else if (event.type === 'done') {
        return event.fullText || fullText
      } else if (event.type === 'error') {
        throw new Error(event.message)
      }
    }
  }

  return fullText
}

export function streamDesign(message, onDelta) {
  return consumeSSE('/api/design', { message }, onDelta)
}

export function streamChat(message, history, params, onDelta) {
  return consumeSSE('/api/chat', { message, history, params }, onDelta)
}

export async function postJson(url, body) {
  const res = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`)
  return data
}

export async function downloadStl(design, name = 'candidate') {
  const res = await fetch('/api/export-stl', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ design, name }),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.error || `HTTP ${res.status}`)
  }
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `${name}.stl`
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}
