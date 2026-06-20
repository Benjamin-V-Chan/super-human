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
      try {
        const event = JSON.parse(line.slice(6))
        if (event.type === 'delta') {
          onDelta(event.text)
          fullText += event.text
        } else if (event.type === 'done') {
          return event.fullText || fullText
        } else if (event.type === 'error') {
          throw new Error(event.message)
        }
      } catch { /* skip malformed */ }
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
