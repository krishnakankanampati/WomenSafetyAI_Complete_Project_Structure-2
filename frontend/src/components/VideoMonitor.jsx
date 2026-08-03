import { useEffect, useMemo, useState } from 'react'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? (import.meta.env.DEV ? "http://localhost:8000" : "");

const CATEGORIES = [
  { label: "Threat", color: "var(--cat-threat)" },
  { label: "Sexual Harassment", color: "var(--cat-sexual)" },
  { label: "Hate Speech", color: "var(--cat-hate)" },
  { label: "Offensive", color: "var(--cat-offensive)" },
];

// Instagram/Facebook share Meta's ~200 calls/user/hour budget across every
// connected item, so they poll slower than YouTube's dedicated quota allows.
const POLL_INTERVAL_MS = {
  youtube: 5000,
  instagram: 30000,
  facebook: 30000,
}

function VideoMonitor({ platform, contentId, onBack, onLogout }) {
  const [incidents, setIncidents] = useState([])

  useEffect(() => {
    fetch(`${BACKEND_URL}/api/monitor/${platform}/${contentId}/start`, { method: "POST" })

    const fetchIncidents = () => {
      fetch(`${BACKEND_URL}/api/incidents/${platform}/${contentId}`)
        .then((res) => res.json())
        .then((data) => setIncidents(data.incidents))
    }
    fetchIncidents()
    const intervalId = setInterval(fetchIncidents, POLL_INTERVAL_MS[platform])

    return () => {
      clearInterval(intervalId)
      fetch(`${BACKEND_URL}/api/monitor/${platform}/${contentId}/stop`, { method: "POST" })
    }
  }, [platform, contentId])

  const counts = useMemo(() => {
    const byLabel = Object.fromEntries(CATEGORIES.map((c) => [c.label, 0]))
    for (const incident of incidents) {
      if (byLabel[incident.label] !== undefined) byLabel[incident.label] += 1
    }
    return byLabel
  }, [incidents])

  const maxCount = Math.max(1, ...CATEGORIES.map((c) => counts[c.label]))

  return (
    <div className="app-shell">
      <div className="monitor-header">
        <button className="icon-btn" onClick={onBack}>← Back</button>
        <h1>{contentId}</h1>
        <span className="live-dot">Live</span>
        <button className="icon-btn" onClick={onLogout}>Logout</button>
      </div>
      <p style={{ color: "var(--text-secondary)", fontSize: 14, marginBottom: 8 }}>
        Checking for new comments every {POLL_INTERVAL_MS[platform] / 1000} seconds. Safe comments are ignored.
      </p>

      <div className="section-label">Analytics</div>
      <div className="stat-grid">
        <div className="stat-tile">
          <div className="stat-tile-value">{incidents.length}</div>
          <div className="stat-tile-label">Total flagged</div>
        </div>
        {CATEGORIES.map((c) => (
          <div className="stat-tile" key={c.label}>
            <div className="stat-tile-value">{counts[c.label]}</div>
            <div className="stat-tile-label">
              <span className="stat-swatch" style={{ background: c.color }} />
              {c.label}
            </div>
          </div>
        ))}
      </div>

      <div className="bar-chart" style={{ marginTop: 12 }}>
        {CATEGORIES.map((c) => (
          <div className="bar-row" key={c.label}>
            <div className="bar-row-label">
              <span className="stat-swatch" style={{ background: c.color }} />
              {c.label}
            </div>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ width: `${(counts[c.label] / maxCount) * 100}%`, background: c.color }}
              />
            </div>
            <div className="bar-row-count">{counts[c.label]}</div>
          </div>
        ))}
      </div>

      <div className="section-label">Live Comment Feed</div>
      {incidents.length === 0 ? (
        <p className="empty-note">No incidents detected yet.</p>
      ) : (
        <div className="incident-list">
          {incidents.map((incident) => {
            const color = CATEGORIES.find((c) => c.label === incident.label)?.color || "var(--text-muted)"
            return (
              <div className="incident-card" key={incident.id} style={{ "--incident-color": color }}>
                <div className="incident-top">
                  <span className="incident-badge">{incident.label}</span>
                  <span className="incident-user">{incident.username}</span>
                </div>
                <div className="incident-comment">"{incident.comment}"</div>
                {incident.llm_reasoning && (
                  <div className="incident-reasoning">{incident.llm_reasoning}</div>
                )}
                <div className="incident-meta">
                  Posted: {new Date(incident.comment_time).toLocaleString()}
                  {" · "}
                  Detected: {new Date(incident.detected_at).toLocaleString()}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default VideoMonitor
