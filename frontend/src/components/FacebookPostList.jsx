import { useEffect, useState } from 'react'
import FacebookIcon from './FacebookIcon'

const BACKEND_URL = "http://localhost:8000";

function formatDate(iso) {
  return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

function FacebookPostList({ onSelectPost, onBack, onLogout }) {
  const [posts, setPosts] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${BACKEND_URL}/api/facebook/posts`)
      .then((res) => res.json())
      .then((data) => setPosts(data.posts))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="app-shell">
      <div className="top-bar">
        <button className="icon-btn" onClick={onBack}>← Platforms</button>
        <span className="yt-logo"><FacebookIcon /></span>
        <h1 style={{ flex: 1 }}>Your Facebook Page Posts</h1>
        <button className="icon-btn" onClick={onLogout}>Logout</button>
      </div>

      {loading ? (
        <p className="center-note">Loading your posts...</p>
      ) : posts.length === 0 ? (
        <p className="empty-note">No posts found on this Page.</p>
      ) : (
        <div className="video-grid">
          {posts.map((p) => (
            <div key={p.id} className="video-card" onClick={() => onSelectPost(p.id)}>
              <div className="video-thumb-wrap fb-thumb-placeholder">
                <FacebookIcon size={40} />
              </div>
              <div className="video-card-body">
                <div className="video-card-title">{p.message || "(No text)"}</div>
                <div className="video-card-sub">{formatDate(p.created_time)}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default FacebookPostList
