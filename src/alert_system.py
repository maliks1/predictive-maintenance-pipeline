"""
Alert System - Binary Alert via HTML Email
Sends notification ONLY when tool needs replacement (status = ALERT).
According to plan.md: Binary Alert System (NORMAL / ALERT).
"""
import os
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()

def build_html_report(prediction: dict, status: str) -> str:
    """
    Build HTML report for binary status (NORMAL / ALERT).

    Args:
        prediction: dict containing keys:
            - vb_mm: predicted flank wear (mm)
            - probability: failure probability (0-1)
            - threshold_mm: critical threshold
            - alert_prob_threshold: alert probability threshold
            - top_features: dict feature_name -> value
        status: "NORMAL" | "ALERT"

    Returns:
        HTML string ready to send.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    is_alert = status == "ALERT"
    color = "#e53e3e" if is_alert else "#48bb78"
    emoji = "🚨" if is_alert else "✅"

    vb = prediction.get("vb_mm", 0)
    prob = prediction.get("probability", 0)
    thresh_mm = prediction.get("threshold_mm", 0.4)
    thresh_prob = prediction.get("alert_prob_threshold", 0.5)
    top_features = prediction.get("top_features", {})

    # Build top features list (up to 5)
    features_html = ""
    if top_features:
        items = sorted(top_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        features_html = "<ul style='padding-left: 20px; color: #4a5568; margin: 0;'>"
        for name, val in items:
            features_html += f"<li style='margin: 4px 0;'><code>{name}</code>: <b>{val:.4f}</b></li>"
        features_html += "</ul>"
    else:
        features_html = "<p style='color: #a0aec0;'>No feature data available.</p>"

    recommendation = (
        "STOP machine immediately. Replace tool before next operation."
        if is_alert
        else "Continue normal operation. Tool condition is within safe limits."
    )

    html = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                 background: #f7fafc; padding: 20px; margin: 0;">
        <div style="max-width: 600px; margin: 0 auto; background: white;
                    border-radius: 12px; overflow: hidden;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.07);">

            <!-- Header -->
            <div style="background: {color}; color: white; padding: 24px;">
                <h1 style="margin: 0; font-size: 24px;">
                    {emoji} Tool Wear Status: {status}
                </h1>
                <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 14px;">
                    {timestamp}
                </p>
            </div>

            <!-- Body -->
            <div style="padding: 24px;">
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                    <tr>
                        <td style="padding: 12px 0; color: #718096; border-bottom: 1px solid #e2e8f0;">
                            Predicted Flank Wear (VB)
                        </td>
                        <td style="padding: 12px 0; text-align: right; font-weight: bold;
                                   font-size: 18px; border-bottom: 1px solid #e2e8f0;
                                   color: {'#e53e3e' if vb > thresh_mm else '#2d3748'};">
                            {vb:.4f} mm
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 0; color: #718096; border-bottom: 1px solid #e2e8f0;">
                            Critical Threshold
                        </td>
                        <td style="padding: 12px 0; text-align: right; border-bottom: 1px solid #e2e8f0;">
                            {thresh_mm:.2f} mm
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 0; color: #718096; border-bottom: 1px solid #e2e8f0;">
                            Failure Probability
                        </td>
                        <td style="padding: 12px 0; text-align: right; font-weight: bold;
                                   font-size: 18px; border-bottom: 1px solid #e2e8f0;
                                   color: {'#e53e3e' if prob >= thresh_prob else '#2d3748'};">
                            {prob:.1%}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 0; color: #718096; border-bottom: 1px solid #e2e8f0;">
                            Alert Threshold
                        </td>
                        <td style="padding: 12px 0; text-align: right; border-bottom: 1px solid #e2e8f0;">
                            {thresh_prob:.0%}
                        </td>
                    </tr>
                </table>

                <h3 style="margin: 0 0 12px 0; color: #2d3748; font-size: 16px;">
                    📊 Top Contributing Features
                </h3>
                {features_html}

                <div style="margin-top: 24px; padding: 16px; background: #f7fafc;
                            border-radius: 8px; border-left: 4px solid {color};">
                    <p style="margin: 0; color: #4a5568; font-size: 14px;">
                        <b>Recommendation:</b> {recommendation}
                    </p>
                </div>
            </div>

            <!-- Footer -->
            <div style="padding: 16px 24px; background: #f7fafc; border-top: 1px solid #e2e8f0;">
                <p style="margin: 0; color: #a0aec0; font-size: 12px; text-align: center;">
                    🤖 Generated by Predictive Maintenance Pipeline<br>
                    Auto-generated alert — do not reply to this email
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    return html


def send_email_alert(prediction: dict, status: str) -> bool:
    """
    Send email alert ONLY when status = ALERT (binary system).

    Args:
        prediction: prediction result dict (see build_html_report)
        status: "NORMAL" | "ALERT"

    Returns:
        True if email was sent, False if skipped/failed.
    """
    # Binary logic: only ALERT requires email
    if status != "ALERT":
        print(f"[INFO] Status is {status}. No email sent.")
        return False

    sender_email = os.getenv("SMTP_EMAIL")
    sender_password = os.getenv("SMTP_PASSWORD")
    receiver_email = os.getenv("ALERT_EMAIL_TO")

    if not all([sender_email, sender_password, receiver_email]):
        print("[WARNING] Email credentials not set. Skipping email alert.")
        return False

    # Build subject
    vb = prediction.get("vb_mm", 0)
    subject = f"[CRITICAL] Tool Wear Alert — VB: {vb:.3f} mm | Status: {status}"

    # Build HTML body
    html_body = build_html_report(prediction, status)

    # Setup message
    msg = MIMEMultipart("alternative")
    msg["From"] = sender_email or ""
    msg["To"] = receiver_email or ""
    msg["Subject"] = subject or ""

    # Plain text fallback
    plain_text = (
        f"Tool Wear Status: {status}\n"
        f"Predicted VB: {vb:.4f} mm\n"
        f"Failure Probability: {prediction.get('probability', 0):.1%}\n"
        f"Timestamp: {datetime.now(tz=timezone.utc).isoformat()}\n\n"
        f"Recommendation: STOP machine immediately. Replace tool before next operation."
    )
    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    # Send
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email or "", sender_password or "")
            server.send_message(msg)
        print(f"[INFO] Alert email sent to {receiver_email} (status: {status})")
        return True
    except (smtplib.SMTPException, TimeoutError, ConnectionError) as e:
        print(f"[ERROR] Failed to send email: {e}")
        return False
