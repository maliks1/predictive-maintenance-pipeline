from fpdf import FPDF
from pathlib import Path
from datetime import datetime

def generate_pdf_report(data: dict, output_dir: Path) -> Path:
    """
    data: dict containing prediction results, example:
    {
        "timestamp": "...", "predicted_vb": 0.24, "threshold": 0.18,
        "probability": 0.85, "status": "CRITICAL", "run_id": "1_1"
    }
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"alert_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = output_dir / filename

    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "PREDICTIVE MAINTENANCE ALERT", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Generated: {data.get('timestamp', datetime.now().isoformat())}", ln=True, align="C")
    pdf.ln(10)

    # Status Box
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(255, 0, 0) if data["status"] == "CRITICAL" else pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Machine Status: {data['status']}", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # Details
    pdf.set_font("Helvetica", "", 12)
    details = [
        ("Run ID", str(data.get("run_id", "N/A"))),
        ("Predicted Tool Wear (VB)", f"{data['predicted_vb']:.4f} mm"),
        ("Critical Threshold", f"{data['threshold']:.2f} mm"),
        ("Failure Probability", f"{data['probability']*100:.2f}%"),
    ]
    
    for label, value in details:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(90, 8, label)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, value, ln=True)

    pdf.ln(10)
    
    # Recommendation
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Recommendation:", ln=True)
    pdf.set_font("Helvetica", "", 12)
    
    if data["status"] == "CRITICAL":
        rec = "Immediately stop the machine and perform tool replacement. Perform thorough inspection."
    elif data["status"] == "WARNING":
        rec = "Monitor machine condition closely. Prepare replacement tools for scheduled replacement."
    else:
        rec = "Machine is operating in normal condition. Continue routine monitoring."
        
    pdf.multi_cell(0, 6, rec)

    pdf.output(str(pdf_path))
    return pdf_path