import { useEffect, useState } from 'react'
import './App.css'
import SiteLogin from './components/SiteLogin'
import PlatformHub from './components/PlatformHub'
import VideoList from './components/VideoList'
import InstagramMediaList from './components/InstagramMediaList'
import FacebookPostList from './components/FacebookPostList'
import VideoMonitor from './components/VideoMonitor'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? (import.meta.env.DEV ? "http://localhost:8000" : "");
const INACTIVITY_LIMIT_MS = 3 * 60 * 1000;

function App() {
  const [checking, setChecking] = useState(true)
  const [youtubeLoggedIn, setYoutubeLoggedIn] = useState(false)
  const [instagramLoggedIn, setInstagramLoggedIn] = useState(false)
  const [facebookLoggedIn, setFacebookLoggedIn] = useState(false)
  const [selectedPlatform, setSelectedPlatform] = useState(null)
  const [selectedContentId, setSelectedContentId] = useState(null)

  useEffect(() => {
    // Same "always show the hub on a fresh visit" requirement as before,
    // now split across two independent session markers: Google's callback
    // has no ?platform param, Meta's carries platform=meta (see
    // backend/main.py's auth_callback / auth_meta_callback redirects).
    const params = new URLSearchParams(window.location.search)
    if (params.get('login') === 'success') {
      const marker = params.get('platform') === 'meta' ? 'loggedInThisSession_meta' : 'loggedInThisSession_youtube'
      sessionStorage.setItem(marker, 'true')
      window.history.replaceState({}, '', window.location.pathname)
    }

    const checks = []

    if (sessionStorage.getItem('loggedInThisSession_youtube') === 'true') {
      checks.push(
        fetch(`${BACKEND_URL}/auth/status`)
          .then((res) => res.json())
          .then((data) => setYoutubeLoggedIn(data.logged_in))
          .catch(() => setYoutubeLoggedIn(false))
      )
    }

    // Deliberately not gated behind a session marker the way YouTube is:
    // Meta access comes from a Page token configured in .env, so there's no
    // login step to re-require on a fresh visit - the backend either has the
    // token or it doesn't. See backend/meta_oauth_store's
    // _credentials_from_env_token for why OAuth isn't used here.
    checks.push(
      fetch(`${BACKEND_URL}/auth/instagram/status`)
        .then((res) => res.json())
        .then((data) => setInstagramLoggedIn(data.logged_in))
        .catch(() => setInstagramLoggedIn(false))
    )
    checks.push(
      fetch(`${BACKEND_URL}/auth/facebook/status`)
        .then((res) => res.json())
        .then((data) => setFacebookLoggedIn(data.logged_in))
        .catch(() => setFacebookLoggedIn(false))
    )

    Promise.all(checks).finally(() => setChecking(false))
  }, [])

  // Ties the browser's own Back button to in-app navigation across the two
  // levels below the hub (platform's list -> that platform's monitor).
  useEffect(() => {
    const onPopState = (e) => {
      const state = e.state || {}
      setSelectedPlatform(state.platform ?? null)
      setSelectedContentId(state.contentId ?? null)
    }
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  const selectPlatform = (platform) => {
    window.history.pushState({ platform, contentId: null }, '')
    setSelectedPlatform(platform)
    setSelectedContentId(null)
  }

  const selectContent = (contentId) => {
    window.history.pushState({ platform: selectedPlatform, contentId }, '')
    setSelectedContentId(contentId)
  }

  const goBack = () => window.history.back()

  // Only YouTube has a session to end. Meta access comes from a .env Page
  // token, so there's nothing to revoke client-side - clearing the Mongo doc
  // wouldn't change what /auth/*/status reports, which would make a "logged
  // out" Instagram/Facebook card silently stay usable. Left out rather than
  // faked.
  const logout = () => {
    fetch(`${BACKEND_URL}/auth/logout`, { method: "POST" }).catch(() => {})
    sessionStorage.removeItem('loggedInThisSession_youtube')
    setYoutubeLoggedIn(false)
    setSelectedPlatform(null)
    setSelectedContentId(null)
  }

  const anyLoggedIn = youtubeLoggedIn

  // Auto-logout after 3 minutes with no mouse/keyboard/scroll activity.
  useEffect(() => {
    if (!anyLoggedIn) return

    let timer = setTimeout(logout, INACTIVITY_LIMIT_MS)
    const resetTimer = () => {
      clearTimeout(timer)
      timer = setTimeout(logout, INACTIVITY_LIMIT_MS)
    }
    const events = ['mousemove', 'keydown', 'click', 'scroll']
    events.forEach((e) => window.addEventListener(e, resetTimer))

    return () => {
      clearTimeout(timer)
      events.forEach((e) => window.removeEventListener(e, resetTimer))
    }
  }, [anyLoggedIn])

  if (checking) {
    return <p className="center-note">Checking login status...</p>
  }

  // Google sign-in is the gate for the whole dashboard, not just the YouTube
  // card - reaching PlatformHub at all now implies youtubeLoggedIn is true,
  // so its YouTube card never renders a "Connect" state in practice.
  if (!youtubeLoggedIn) {
    return <SiteLogin />
  }

  if (!selectedPlatform) {
    return (
      <PlatformHub
        youtubeLoggedIn={youtubeLoggedIn}
        instagramLoggedIn={instagramLoggedIn}
        facebookLoggedIn={facebookLoggedIn}
        onSelectPlatform={selectPlatform}
        onLogout={anyLoggedIn ? logout : undefined}
      />
    )
  }

  if (!selectedContentId) {
    if (selectedPlatform === 'youtube') {
      return <VideoList onSelectVideo={selectContent} onBack={goBack} onLogout={logout} />
    }
    if (selectedPlatform === 'instagram') {
      return <InstagramMediaList onSelectMedia={selectContent} onBack={goBack} onLogout={logout} />
    }
    return <FacebookPostList onSelectPost={selectContent} onBack={goBack} onLogout={logout} />
  }

  return (
    <VideoMonitor
      platform={selectedPlatform}
      contentId={selectedContentId}
      onBack={goBack}
      onLogout={logout}
    />
  )
}

export default App
