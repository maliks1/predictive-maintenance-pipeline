import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path

def send_email_alert(subject: str, body: str, attachment_path: Path):
    # Konfigurasi dari Environment Variables
    sender_email = os.getenv("SMTP_EMAIL")
    sender_password = os.getenv("SMTP_PASSWORD") # Gunakan App Password
    receiver_email = os.getenv("ALERT_EMAIL_TO")
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    if not all([sender_email, sender_password, receiver_email]):
        raise ValueError("Environment variables untuk email (SMTP_EMAIL, SMTP_PASSWORD, ALERT_EMAIL_TO) belum diset.")

    # Setup Pesan
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

    # Kirim Email
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print(f"[INFO] Alert email berhasil dikirim ke {receiver_email}")
    except Exception as e:
        print(f"[ERROR] Gagal mengirim email: {e}")
        raise