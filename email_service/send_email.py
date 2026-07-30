"""
Sends an alert email to the NGO contact when a non-Safe comment is detected.
Uses Gmail SMTP with an App Password (see .env.example for setup).
"""

import smtplib
from email.message import EmailMessage

from email_service.config import (
    NGO_EMAIL,
    SENDER_APP_PASSWORD,
    SENDER_EMAIL,
    SMTP_HOST,
    SMTP_PORT,
)

TEXT_TEMPLATE = """Hello,

Our automated monitoring system has detected a potentially harmful comment
that requires your attention.

Platform     : {platform}
Video ID     : {video_id}
Username     : {username}
Comment      : "{comment}"
Category     : {label}
Posted At    : {comment_time}
Detected At  : {detected_at}

AI Reasoning : {reasoning}

Please review this comment and take appropriate action.

Women Safety AI Monitoring System (automated alert)
"""

HTML_TEMPLATE = """\
<html>
  <body style="font-family: Arial, Helvetica, sans-serif; font-size: 14px; color: #1a1a1a;">
    <p>Hello,</p>
    <p>Our automated monitoring system has detected a potentially harmful comment
    that requires your attention.</p>
    <table cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
      <tr><td><strong>Platform</strong></td><td><strong>{platform}</strong></td></tr>
      <tr><td><strong>Video ID</strong></td><td><strong>{video_id}</strong></td></tr>
      <tr><td><strong>Username</strong></td><td><strong>{username}</strong></td></tr>
      <tr><td><strong>Comment</strong></td><td><strong>"{comment}"</strong></td></tr>
      <tr><td><strong>Category</strong></td><td><strong>{label}</strong></td></tr>
      <tr><td><strong>Posted At</strong></td><td><strong>{comment_time}</strong></td></tr>
      <tr><td><strong>Detected At</strong></td><td><strong>{detected_at}</strong></td></tr>
    </table>
    <p><strong>AI Reasoning:</strong> {reasoning}</p>
    <p>Please review this comment and take appropriate action.</p>
    <p>Women Safety AI Monitoring System (automated alert)</p>
  </body>
</html>
"""


def send_alert_email(platform, video_id, username, comment, comment_time, label, confidence, detected_at, llm_reasoning=None):
    if not (SENDER_EMAIL and SENDER_APP_PASSWORD):
        print("Email alert skipped: SENDER_EMAIL/SENDER_APP_PASSWORD not set in .env.")
        return False

    reasoning = llm_reasoning or "N/A (Gemini reasoning unavailable - check GEMINI_API_KEY/quota)"
    fields = dict(
        platform=platform,
        video_id=video_id,
        username=username,
        comment=comment,
        label=label,
        comment_time=comment_time,
        detected_at=detected_at,
        reasoning=reasoning,
    )

    msg = EmailMessage()
    msg["Subject"] = f"[Women Safety Alert] {label} detected on {platform}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = NGO_EMAIL
    msg.set_content(TEXT_TEMPLATE.format(**fields))
    msg.add_alternative(HTML_TEMPLATE.format(**fields), subtype="html")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        server.send_message(msg)

    return True
