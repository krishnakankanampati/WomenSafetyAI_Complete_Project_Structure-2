import YouTubeIcon from './YouTubeIcon'
import InstagramIcon from './InstagramIcon'
import FacebookIcon from './FacebookIcon'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? (import.meta.env.DEV ? "http://localhost:8000" : "");

function ShieldIcon() {
  return (
    <svg width="46" height="46" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 2 4 5v6c0 5 3.4 9.4 8 11 4.6-1.6 8-6 8-11V5l-8-3z"
        fill="#fff"
        opacity="0.18"
      />
      <path
        d="M12 2 4 5v6c0 5 3.4 9.4 8 11 4.6-1.6 8-6 8-11V5l-8-3z"
        stroke="#fff"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
      <path
        d="M8.5 12.2 11 14.7 15.8 9.5"
        stroke="#fff"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function GoogleGlyph() {
  return (
    <svg width="20" height="20" viewBox="0 0 18 18" aria-hidden="true">
      <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84c-.21 1.13-.84 2.08-1.8 2.72v2.26h2.92c1.7-1.57 2.68-3.88 2.68-6.62z" />
      <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.81.54-1.84.86-3.04.86-2.34 0-4.32-1.58-5.03-3.7H.96v2.33C2.44 15.98 5.48 18 9 18z" />
      <path fill="#FBBC05" d="M3.97 10.72c-.18-.54-.28-1.11-.28-1.72s.1-1.18.28-1.72V4.95H.96A8.996 8.996 0 0 0 0 9c0 1.45.35 2.83.96 4.05l3.01-2.33z" />
      <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58C13.46.89 11.43 0 9 0 5.48 0 2.44 2.02.96 4.95l3.01 2.33C4.68 5.16 6.66 3.58 9 3.58z" />
    </svg>
  );
}

function SiteLogin() {
  return (
    <div className="hero-screen">
      <div className="hero-orb hero-orb-a" />
      <div className="hero-orb hero-orb-b" />
      <div className="hero-orb hero-orb-c" />

      <div className="hero-content">
        <div className="hero-badge">
          <span className="hero-badge-ring" />
          <ShieldIcon />
        </div>

        <h1 className="hero-title">Women Safety AI</h1>
        <p className="hero-tagline">
          Real-time AI moderation that watches your comments so you don't have to.
        </p>

        <div className="hero-platform-row">
          <span className="hero-platform-chip"><YouTubeIcon size={18} /> YouTube</span>
          <span className="hero-platform-chip"><InstagramIcon size={18} /> Instagram</span>
          <span className="hero-platform-chip"><FacebookIcon size={18} /> Facebook</span>
        </div>

        <button
          className="hero-google-btn"
          onClick={() => { window.location.href = `${BACKEND_URL}/auth/login`; }}
        >
          <GoogleGlyph />
          Continue with Google
        </button>

        <p className="hero-footnote">Only you can sign in. Your videos and comments stay private.</p>
      </div>
    </div>
  );
}

export default SiteLogin;
