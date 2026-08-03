import { useEffect, useState } from 'react'
import InstagramIcon from './InstagramIcon'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? (import.meta.env.DEV ? "http://localhost:8000" : "");

function formatDate(iso) {
  return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

function InstagramMediaList({ onSelectMedia, onBack, onLogout }) {
  const [media, setMedia] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${BACKEND_URL}/api/instagram/media`)
      .then((res) => res.json())
      .then((data) => setMedia(data.media))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="app-shell">
      <div className="top-bar">
        <button className="icon-btn" onClick={onBack}>← Platforms</button>
        <span className="yt-logo"><InstagramIcon /></span>
        <h1 style={{ flex: 1 }}>Your Instagram Posts</h1>
        <button className="icon-btn" onClick={onLogout}>Logout</button>
      </div>

      {loading ? (
        <p className="center-note">Loading your posts...</p>
      ) : media.length === 0 ? (
        <p className="empty-note">No posts found on this account.</p>
      ) : (
        <div className="video-grid">
          {media.map((m) => (
            <div key={m.id} className="video-card" onClick={() => onSelectMedia(m.id)}>
              <div className="video-thumb-wrap">
                <img src={m.thumbnail_url || m.media_url} alt="" />
              </div>
              <div className="video-card-body">
                <div className="video-card-title">{m.caption || "(No caption)"}</div>
                <div className="video-card-sub">{formatDate(m.timestamp)}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default InstagramMediaList
