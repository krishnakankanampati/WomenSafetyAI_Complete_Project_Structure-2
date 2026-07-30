function InstagramIcon({ size = 28 }) {
  const gradientId = "ig-gradient"
  return (
    <svg width={size} height={size} viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <defs>
        <linearGradient id={gradientId} x1="0%" y1="100%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#FFDD55" />
          <stop offset="35%" stopColor="#FF543E" />
          <stop offset="70%" stopColor="#C837AB" />
          <stop offset="100%" stopColor="#5C5EDD" />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="26" height="26" rx="7" fill={`url(#${gradientId})`} />
      <rect x="7.5" y="7.5" width="13" height="13" rx="4.5" fill="none" stroke="#fff" strokeWidth="1.8" />
      <circle cx="14" cy="14" r="3.6" fill="none" stroke="#fff" strokeWidth="1.8" />
      <circle cx="19.2" cy="8.8" r="1.1" fill="#fff" />
    </svg>
  );
}

export default InstagramIcon;
