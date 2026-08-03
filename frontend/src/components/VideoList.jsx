import { useEffect, useState } from 'react'
import YouTubeIcon from './YouTubeIcon'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? (import.meta.env.DEV ? "http://localhost:8000" : "");

function formatDate(iso) {
  return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

function VideoList({ onSelectVideo, onBack, onLogout }) {
  const [videos, setVideos] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${BACKEND_URL}/api/videos`)
      .then((res) => res.json())
      .then((data) => setVideos(data.videos))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="app-shell">
      <div className="top-bar">
        <button className="icon-btn" onClick={onBack}>← Platforms</button>
        <span className="yt-logo"><YouTubeIcon /></span>
        <h1 style={{ flex: 1 }}>Your YouTube Videos</h1>
        <button className="icon-btn" onClick={onLogout}>Logout</button>
      </div>

      {loading ? (
        <p className="center-note">Loading your videos...</p>
      ) : videos.length === 0 ? (
        <p className="empty-note">No videos found on this channel.</p>
      ) : (
        <div className="video-grid">
          {videos.map((v) => (
            <div key={v.video_id} className="video-card" onClick={() => onSelectVideo(v.video_id)}>
              <div className="video-thumb-wrap">
                <img src={v.thumbnail_url} alt="" />
              </div>
              <div className="video-card-body">
                <div className="video-card-title">{v.title}</div>
                <div className="video-card-sub">{formatDate(v.published_at)}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default VideoList
