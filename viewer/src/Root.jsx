import { useEffect, useState } from 'react'
import DemoSingle from './demo/DemoSingle.jsx'
import DemoPage from './demo/DemoPage.jsx'
import LandingPage from './landing/LandingPage.jsx'

// Hash router:
//   (default) → landing page
//   #demo     → single-clip, fully client-side pipeline (Vercel-deployable)
//   #lab      → multi-clip Python pipeline (needs the Python backend host)
export default function Root() {
  const [hash, setHash] = useState(window.location.hash)
  useEffect(() => {
    const onHash = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  if (hash === '#lab') return <DemoPage />
  if (hash === '#demo') return <DemoSingle />
  return <LandingPage />
}
