import { useEffect, useState } from 'react'
import App from './App.jsx'
import DemoPage from './demo/DemoPage.jsx'

// Lightweight hash router: '#studio' → the design studio, anything else → the
// multi-agent pipeline demo (the default landing experience).
export default function Root() {
  const [hash, setHash] = useState(window.location.hash)
  useEffect(() => {
    const onHash = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  return hash === '#studio' ? <App /> : <DemoPage />
}
