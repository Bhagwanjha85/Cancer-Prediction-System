import os
import csv
import logging
import datetime
import torch
from typing import Dict, Any, List
from io import BytesIO

# Config logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log"))
    ]
)
logger = logging.getLogger("CancerDetectionSystem")

def check_gpu_support() -> Dict[str, Any]:
    """
    Checks GPU availability and configurations in PyTorch.
    """
    gpu_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if gpu_available else 0
    devices = [torch.cuda.get_device_name(i) for i in range(device_count)]
    
    support_info = {
        "gpu_available": gpu_available,
        "device_count": device_count,
        "devices": devices,
        "active_device": f"GPU: {devices[0]}" if gpu_available else "CPU (Fallback)"
    }
    
    if gpu_available:
        logger.info(f"GPU Support detected: {devices}")
    else:
        logger.info("No GPU detected. Running on CPU Fallback.")
        
    return support_info

def append_prediction_to_csv(csv_path: str, record: Dict[str, Any]) -> None:
    """
    Appends a new prediction entry to a local CSV history file.
    """
    file_exists = os.path.exists(csv_path)
    
    headers = [
        "timestamp", 
        "detection_type", 
        "predicted_class", 
        "confidence", 
        "is_cancerous", 
        "device_used"
    ]
    
    try:
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": record.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "detection_type": record.get("detection_type", "Unknown"),
                "predicted_class": record.get("predicted_class", "Unknown"),
                "confidence": f"{record.get('confidence', 0.0):.4f}",
                "is_cancerous": str(record.get("is_cancerous", False)),
                "device_used": record.get("device_used", "CPU")
            })
    except Exception as e:
        logger.error(f"Failed to write prediction history to CSV: {str(e)}")

def generate_pdf_report(record: Dict[str, Any]) -> bytes:
    """
    Generates a professional PDF report containing the prediction results,
    disease description, medical advice, and disclaimer using ReportLab.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    except ImportError:
        logger.warning("ReportLab not installed. Generating dummy text PDF fallback.")
        # Minimal fallback if reportlab is missing
        buffer = BytesIO()
        buffer.write(b"PDF Generation Fallback: ReportLab is not installed.\n")
        buffer.write(f"Prediction Record:\n{str(record)}".encode('utf-8'))
        return buffer.getvalue()

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=15,
        alignment=1  # Center
    )
    
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748")
    )
    
    disclaimer_style = ParagraphStyle(
        "ReportDisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#E53E3E")
    )
    
    elements = []
    
    # Document Title
    elements.append(Paragraph("AI Cancer Detection Report", title_style))
    elements.append(Spacer(1, 10))
    
    # Patient/Session Metadata Table
    timestamp = record.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    metadata_data = [
        [Paragraph("<b>Report Date:</b>", body_style), Paragraph(timestamp, body_style),
         Paragraph("<b>Portal Version:</b>", body_style), Paragraph("v1.0.0", body_style)],
        [Paragraph("<b>Selected Category:</b>", body_style), Paragraph(record.get("detection_type", "N/A"), body_style),
         Paragraph("<b>Analysis Device:</b>", body_style), Paragraph(record.get("device_used", "CPU"), body_style)]
    ]
    
    meta_table = Table(metadata_data, colWidths=[120, 140, 120, 140])
    meta_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    
    elements.append(meta_table)
    elements.append(Spacer(1, 15))
    
    # Primary Diagnosis Result
    pred_class = record.get("predicted_class", "Unknown").replace("_", " ")
    conf = record.get("confidence", 0.0)
    is_cancer = record.get("is_cancerous", False)
    
    result_bg = "#FED7D7" if is_cancer else "#C6F6D5"
    result_text_color = "#9B2C2C" if is_cancer else "#22543D"
    
    result_data = [
        [Paragraph("<b>PREDICTION ANALYSIS SUMMARY</b>", ParagraphStyle("SummaryHeader", parent=body_style, fontSize=11, fontName="Helvetica-Bold", textColor=colors.HexColor(result_text_color)))],
        [Paragraph(f"<b>Classification:</b> {pred_class}", body_style)],
        [Paragraph(f"<b>Confidence Level:</b> {conf*100:.2f}%", body_style)],
        [Paragraph(f"<b>Status:</b> {'Malignancy Suspected' if is_cancer else 'No Malignancy Detected'} ", body_style)]
    ]
    
    result_table = Table(result_data, colWidths=[520])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(result_bg)),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor(result_text_color)),
        ('PADDING', (0,0), (-1,-1), 12),
    ]))
    
    elements.append(result_table)
    elements.append(Spacer(1, 15))
    
    # Disease Info
    info = record.get("medical_info", {})
    if info:
        elements.append(Paragraph("Clinical Description", section_style))
        elements.append(Paragraph(info.get("description", "No description available."), body_style))
        elements.append(Spacer(1, 10))
        
        elements.append(Paragraph("Typical Symptoms", section_style))
        symptoms_text = "<br/>".join([f"&bull; {s}" for s in info.get("symptoms", [])])
        elements.append(Paragraph(symptoms_text or "No typical symptoms recorded.", body_style))
        elements.append(Spacer(1, 10))
        
        elements.append(Paragraph("Risk Factors", section_style))
        risks_text = "<br/>".join([f"&bull; {r}" for r in info.get("risk_factors", [])])
        elements.append(Paragraph(risks_text or "No specific risk factors listed.", body_style))
        elements.append(Spacer(1, 10))
        
        elements.append(Paragraph("Preventive Measures & Recommendations", section_style))
        prev_text = "<br/>".join([f"&bull; {p}" for p in info.get("preventive_measures", [])])
        elements.append(Paragraph(prev_text or "No preventive measures listed.", body_style))
        elements.append(Spacer(1, 10))
        
        elements.append(Paragraph("Standard Treatment Options", section_style))
        treatments_text = "<br/>".join([f"&bull; {t}" for t in info.get("treatment_suggestions", [])])
        elements.append(Paragraph(treatments_text or "Consult medical specialists for options.", body_style))
        elements.append(Spacer(1, 10))
        
        elements.append(Paragraph("Recommended Doctor Consultation Advice", section_style))
        elements.append(Paragraph(info.get("consultation_advice", "Seek immediate expert evaluation."), body_style))
        elements.append(Spacer(1, 15))
        
    # Divider line
    elements.append(Table([[""]], colWidths=[520], rowHeights=[1], style=TableStyle([('LINEBELOW', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0"))])))
    elements.append(Spacer(1, 10))
    
    # Disclaimer Section
    elements.append(Paragraph("<b>CRITICAL MEDICAL DISCLAIMER:</b>", ParagraphStyle("DisclaimerTitle", parent=disclaimer_style, fontName="Helvetica-Bold")))
    elements.append(Paragraph(record.get("disclaimer", "This AI model is for research and educational purposes only."), disclaimer_style))
    
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes
