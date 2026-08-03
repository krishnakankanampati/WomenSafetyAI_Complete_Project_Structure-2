import YouTubeIcon from './YouTubeIcon'
import InstagramIcon from './InstagramIcon'
import FacebookIcon from './FacebookIcon'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? (import.meta.env.DEV ? "http://localhost:8000" : "");

function ShieldIcon() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 2 4 5v6c0 5 3.4 9.4 8 11 4.6-1.6 8-6 8-11V5l-8-3z" fill="#fff" opacity="0.18" />
      <path d="M12 2 4 5v6c0 5 3.4 9.4 8 11 4.6-1.6 8-6 8-11V5l-8-3z" stroke="#fff" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M8.5 12.2 11 14.7 15.8 9.5" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function PlatformTile({ accent, icon, name, description, state, caveat, onOpen, onConnect }) {
  return (
    <div className="tile" style={{ "--accent": accent }}>
      <div className="tile-glow" />
      <div className="tile-top">
        <div className="tile-icon">{icon}</div>
        <span className={`tile-status tile-status-${state === "usable" ? "on" : "off"}`}>
          <span className="tile-status-dot" />
          {state === "usable" ? "Connected" : "Not connected"}
        </span>
      </div>

      <div className="tile-name">{name}</div>
      <div className="tile-desc">{description}</div>

      <div className="tile-action">
        {state === "usable" && (
          <button className="tile-btn tile-btn-primary" onClick={onOpen}>
            Open dashboard
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
        {state === "connect" && (
          <button className="tile-btn" onClick={onConnect}>Connect</button>
        )}
        {state === "caveat" && (
          <p className="tile-caveat">{caveat}</p>
        )}
      </div>
    </div>
  );
}

function PlatformHub({ youtubeLoggedIn, instagramLoggedIn, facebookLoggedIn, onSelectPlatform, onLogout }) {
  // Instagram and Facebook both come from one Page token in .env, so their
  // "not available" reasons differ: no token at all vs. a token whose Page
  // has no linked Instagram Business account. facebookLoggedIn tells them
  // apart, since any valid token yields at least the Page itself.
  const instagramState = instagramLoggedIn ? "usable" : "caveat"
  const instagramCaveat = facebookLoggedIn
    ? "No linked Instagram Business account"
    : "Set META_PAGE_ACCESS_TOKEN in .env"
  const facebookState = facebookLoggedIn ? "usable" : "caveat"

  return (
    <div className="dash-screen">
      <div className="hero-orb hero-orb-a dash-orb-a" />
      <div className="hero-orb hero-orb-b dash-orb-b" />

      <div className="dash-content">
        <div className="dash-top">
          <div className="dash-brand">
            <span className="dash-brand-badge"><ShieldIcon /></span>
            <div>
              <div className="dash-brand-name">Women Safety AI</div>
              <div className="dash-brand-sub">Choose a platform to monitor</div>
            </div>
          </div>
          <button className="icon-btn dash-logout" onClick={onLogout}>Logout</button>
        </div>

        <div className="tile-grid">
          <PlatformTile
            accent="var(--yt-red)"
            icon={<YouTubeIcon size={32} />}
            name="YouTube"
            description="Live comment monitoring on your videos"
            state={youtubeLoggedIn ? "usable" : "connect"}
            onOpen={() => onSelectPlatform("youtube")}
            onConnect={() => { window.location.href = `${BACKEND_URL}/auth/login`; }}
          />
          <PlatformTile
            accent="#C837AB"
            icon={<InstagramIcon size={32} />}
            name="Instagram"
            description="Live comment monitoring on your posts"
            state={instagramState}
            caveat={instagramCaveat}
            onOpen={() => onSelectPlatform("instagram")}
          />
          <PlatformTile
            accent="#1877F2"
            icon={<FacebookIcon size={32} />}
            name="Facebook"
            description="Live comment monitoring on your Page posts"
            state={facebookState}
            caveat="Set META_PAGE_ACCESS_TOKEN in .env"
            onOpen={() => onSelectPlatform("facebook")}
          />
        </div>

        <p className="dash-footnote">
          Instagram and Facebook share one Page token from .env, so they're always
          available together - only YouTube has a login to sign out of.
        </p>
      </div>
    </div>
  );
}

export default PlatformHub;
