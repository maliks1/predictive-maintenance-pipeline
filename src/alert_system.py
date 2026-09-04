import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


def send_email_alert(subject: str, body: str, attachment_path: Path):
    # Configuration from Environment Variables
    sender_email = os.getenv("SMTP_EMAIL")
    sender_password = os.getenv("SMTP_PASSWORD")  # Use App Password
    receiver_email = os.getenv("ALERT_EMAIL_TO")
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    if not all([sender_email, sender_password, receiver_email]):
        raise ValueError("Environment variables for email (SMTP_EMAIL, SMTP_PASSWORD, ALERT_EMAIL_TO) not set.")

    # Setup Message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    # Attach PDF
    if attachment_path and attachment_path.exists():
        with open(attachment_path, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())

        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f"attachment; filename= {attachment_path.name}"
        )
        msg.attach(part)

    # Send Email
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print(f"[INFO] Alert email successfully sent to {receiver_email}")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        raise
