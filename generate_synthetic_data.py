"""
generate_synthetic_data.py — Generates 20 format-valid synthetic test screenshots
Phase 1 Deliverable — Screen PII Redactor (Snapdragon AI Lab Challenge)

Section 4 Specification:
- Zero real PII.
- 15-25 synthetic mock screenshots:
  1. Mock email client (email, phone, UPI ID in body)
  2. Mock KYC/onboarding form (Aadhaar, PAN fields)
  3. Mock banking dashboard (IFSC, account details)
  4. Mock chat/support ticket (mixed PII types)
  5. 3-5 clean screenshots with NO PII (to check false-positive rate)
- Generates pixel-accurate images and records ground-truth JSON annotations.
"""

import os
import json
import random
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont

from pii_classifier import generate_verhoeff, generate_luhn

OUTPUT_DIR = "synthetic_test_set"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper to get standard fonts safely
def get_font(size: int, bold: bool = False):
    try:
        font_name = "arialbd.ttf" if bold else "arial.ttf"
        return ImageFont.truetype(font_name, size)
    except IOError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except IOError:
            return ImageFont.load_default()

FONT_TITLE = get_font(22, bold=True)
FONT_HEADING = get_font(18, bold=True)
FONT_BODY = get_font(16, bold=False)
FONT_BOLD = get_font(16, bold=True)
FONT_SMALL = get_font(13, bold=False)

def draw_window_frame(draw: ImageDraw.ImageDraw, width: int, height: int, title: str, bg_color=(245, 247, 250)):
    # Canvas background
    draw.rectangle([0, 0, width, height], fill=bg_color)
    # Window title bar (mac/modern style)
    draw.rectangle([0, 0, width, 38], fill=(30, 41, 59))
    # Window controls
    draw.ellipse([14, 12, 26, 24], fill=(239, 68, 68))
    draw.ellipse([34, 12, 46, 24], fill=(245, 158, 11))
    draw.ellipse([54, 12, 66, 24], fill=(16, 185, 129))
    # Window title
    draw.text((85, 10), title, font=FONT_HEADING, fill=(241, 245, 249))

def draw_card(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, fill=(255, 255, 255), outline=(226, 232, 240)):
    draw.rounded_rectangle([x1, y1, x2, y2], radius=8, fill=fill, outline=outline, width=1)


# =====================================================================
# DATA GENERATORS
# =====================================================================

def create_synthetic_datasets() -> List[Dict[str, Any]]:
    test_cases = []
    
    # -------------------------------------------------------------
    # 1. EMAIL CLIENTS (4 images)
    # -------------------------------------------------------------
    email_scenarios = [
        {
            "name": "email_client_01.png",
            "title": "Inbox — MailClient Pro",
            "sender": "arun.kumar92@gmail.com",
            "subject": "Invoice clearance and consulting fee reimbursement",
            "phone": "+91 9845123456",
            "upi": "arunkumar@okhdfcbank",
            "lines": [
                "Dear Accounts Team,",
                "Please find the invoice for the cloud architecture sprint.",
                "Kindly process the reimbursement to my primary UPI ID or contact me:",
                "Primary UPI: arunkumar@okhdfcbank",
                "Direct Mobile: +91 9845123456",
                "Official Email: arun.kumar92@gmail.com",
                "Thanks & Regards,",
                "Arun Kumar S — Lead Cloud Architect"
            ]
        },
        {
            "name": "email_client_02.png",
            "title": "Support Mailbox — Ticket #84910",
            "sender": "support.desk@enterprise-saas.com",
            "subject": "Urgent: Update your billing coordinates",
            "phone": "9876543210",
            "upi": "billing.dept@paytm",
            "lines": [
                "Hello Priya,",
                "Your subscription renewal failed due to banking authorization timeout.",
                "You can initiate instantaneous transfer via UPI:",
                "UPI Payment: billing.dept@paytm",
                "Helpline Support: 9876543210",
                "Escalation Email: support.desk@enterprise-saas.com",
                "Thank you for choosing Enterprise SaaS."
            ]
        },
        {
            "name": "email_client_03.png",
            "title": "Vendor Portal Mail — Settlement Notice",
            "sender": "vendor.settlement@logistics-india.org",
            "subject": "Weekly Vendor Freight Settlement Summary",
            "phone": "+91 7890123456",
            "upi": "logistics.ops@ybl",
            "lines": [
                "Attn: Operations Manager,",
                "The weekly payout for Chennai-Bengaluru corridor has been initiated.",
                "For query resolutions reach out via:",
                "Operations Cell: +91 7890123456",
                "Dispute UPI Handle: logistics.ops@ybl",
                "Official Helpdesk: vendor.settlement@logistics-india.org",
                "Please acknowledge receipt."
            ]
        },
        {
            "name": "email_client_04.png",
            "title": "HR Onboarding Desk — Welcome Package",
            "sender": "hr.connect@techcorp.in",
            "subject": "Candidate Onboarding & Reimbursement Setup",
            "phone": "8765432109",
            "upi": "vikram.aditya@okaxis",
            "lines": [
                "Welcome to the engineering team!",
                "Kindly submit your asset reimbursement details to the desk.",
                "Designated POC Phone: 8765432109",
                "HR Representative Email: hr.connect@techcorp.in",
                "Direct UPI Verification: vikram.aditya@okaxis",
                "Best regards, Human Resources Division"
            ]
        }
    ]
    
    for sc in email_scenarios:
        test_cases.append({
            "category": "EMAIL_CLIENT",
            "file": sc["name"],
            "title": sc["title"],
            "scenario": sc
        })

    # -------------------------------------------------------------
    # 2. KYC / ONBOARDING FORMS (4 images)
    # -------------------------------------------------------------
    kyc_scenarios = [
        {
            "name": "kyc_onboarding_01.png",
            "title": "FinTech Prime — Digital KYC Identity Verification",
            "pan": "ABCDE1234F",
            "aadhaar": generate_verhoeff("98765432101"),  # 987654321012
            "phone": "+91 9123456789",
            "name_val": "Rajesh Ramanathan",
            "dob": "14/08/1990"
        },
        {
            "name": "kyc_onboarding_02.png",
            "title": "National Securities Registry — Demat e-KYC",
            "pan": "BKZPR9876Q",
            "aadhaar": generate_verhoeff("54321678901"),  # 543216789018
            "phone": "9988776655",
            "name_val": "Ananya Mukherjee",
            "dob": "02/11/1988"
        },
        {
            "name": "kyc_onboarding_03.png",
            "title": "Neobank India — Instant Account Creation",
            "pan": "FGHIJ5678K",
            "aadhaar": generate_verhoeff("34567890123"),  # 345678901235
            "phone": "+91 8765432100",
            "name_val": "Siddharth Verma",
            "dob": "25/04/1995"
        },
        {
            "name": "kyc_onboarding_04.png",
            "title": "Home Loan Portal — Borrower Document Verification",
            "pan": "KLMNO9012Z",
            "aadhaar": generate_verhoeff("78901234567"),
            "phone": "7012345678",
            "name_val": "Kavitha Sundaram",
            "dob": "19/09/1984"
        }
    ]
    for sc in kyc_scenarios:
        test_cases.append({
            "category": "KYC_FORM",
            "file": sc["name"],
            "title": sc["title"],
            "scenario": sc
        })

    # -------------------------------------------------------------
    # 3. BANKING DASHBOARDS (4 images)
    # -------------------------------------------------------------
    banking_scenarios = [
        {
            "name": "banking_dashboard_01.png",
            "title": "HDFC NetBanking — Corporate Funds Transfer",
            "ifsc": "HDFC0001234",
            "card": generate_luhn("4532", 16),
            "upi": "treasury.ops@okhdfcbank",
            "email": "corp.treasury@hdfc-client.com",
            "bank_name": "HDFC Bank Ltd, Nariman Point Branch"
        },
        {
            "name": "banking_dashboard_02.png",
            "title": "SBI Corporate Portal — NEFT / RTGS Initiation",
            "ifsc": "SBIN0004321",
            "card": generate_luhn("5241", 16),
            "upi": "merchant.settle@oksbi",
            "email": "settlements@sbi-merchants.in",
            "bank_name": "State Bank of India, MG Road Bangalore"
        },
        {
            "name": "banking_dashboard_03.png",
            "title": "ICICI Bank Infinity — Beneficiary Management",
            "ifsc": "ICIC0000987",
            "card": generate_luhn("4111", 16),
            "upi": "payroll.support@okicici",
            "email": "infinity.help@icicibank.com",
            "bank_name": "ICICI Bank, Bandra Kurla Complex"
        },
        {
            "name": "banking_dashboard_04.png",
            "title": "Axis Direct — Automated Clearing System",
            "ifsc": "UTIB0000456",
            "card": generate_luhn("6011", 16),
            "upi": "direct.payout@axisbank",
            "email": "clearing.house@axis-portal.org",
            "bank_name": "Axis Bank Ltd, Connaught Place New Delhi"
        }
    ]
    for sc in banking_scenarios:
        test_cases.append({
            "category": "BANKING_DASHBOARD",
            "file": sc["name"],
            "title": sc["title"],
            "scenario": sc
        })

    # -------------------------------------------------------------
    # 4. CHAT / SUPPORT TICKETS (4 images)
    # -------------------------------------------------------------
    chat_scenarios = [
        {
            "name": "chat_support_01.png",
            "title": "Slack — #finops-incident-room",
            "messages": [
                ("Lead Ops [10:14 AM]", "We have a dispute for merchant refund settlement."),
                ("Finance Rep [10:15 AM]", "Understood. Please send customer contact and UPI ID."),
                ("Lead Ops [10:16 AM]", "Customer phone is +91 9840123456 and UPI is priya99@okaxis"),
                ("Finance Rep [10:17 AM]", "Got it. Registered email on record: priya.nair@sampledomain.com"),
                ("Finance Rep [10:18 AM]", "Processing reversal via IFSC UTIB0000123.")
            ],
            "pii_items": [
                ("+91 9840123456", "PHONE_IN"),
                ("priya99@okaxis", "UPI_ID"),
                ("priya.nair@sampledomain.com", "EMAIL"),
                ("UTIB0000123", "IFSC")
            ]
        },
        {
            "name": "chat_support_02.png",
            "title": "Zendesk Customer Chat — Ticket #99281",
            "messages": [
                ("Agent Rahul", "Hello! Welcome to E-Commerce Express Support."),
                ("Customer Sneha", "Hi, my courier package is delayed."),
                ("Agent Rahul", "May I verify your delivery phone number?"),
                ("Customer Sneha", "Yes, it is 9444012345 and email is sneha.b@webmail.in"),
                ("Agent Rahul", "Thank you Sneha. Verification complete.")
            ],
            "pii_items": [
                ("9444012345", "PHONE_IN"),
                ("sneha.b@webmail.in", "EMAIL")
            ]
        },
        {
            "name": "chat_support_03.png",
            "title": "Microsoft Teams — Identity Verification Bridge",
            "messages": [
                ("Compliance Officer", "Please confirm candidate PAN and Aadhaar for background check."),
                ("HR Specialist", "Candidate PAN: CDEFG5678H"),
                ("HR Specialist", f"Aadhaar UID: {generate_verhoeff('45678901234')}"),
                ("Compliance Officer", "Verified against NSDL database.")
            ],
            "pii_items": [
                ("CDEFG5678H", "PAN"),
                (generate_verhoeff("45678901234"), "AADHAAR")
            ]
        },
        {
            "name": "chat_support_04.png",
            "title": "Telegram Customer Helpdesk — Payment Dispute",
            "messages": [
                ("Helpdesk Bot", "Please provide billing credentials to reconcile card charge."),
                ("User Deepak", f"Card charged was {generate_luhn('4222', 16)}"),
                ("User Deepak", "My contact number is +91 9820012345"),
                ("Helpdesk Bot", "Reconciliation initiated. Confirmation sent to deepak.k@cloudmail.org")
            ],
            "pii_items": [
                (generate_luhn("4222", 16), "CARD_NUMBER"),
                ("+91 9820012345", "PHONE_IN"),
                ("deepak.k@cloudmail.org", "EMAIL")
            ]
        }
    ]
    for sc in chat_scenarios:
        test_cases.append({
            "category": "CHAT_SUPPORT",
            "file": sc["name"],
            "title": sc["title"],
            "scenario": sc
        })

    # -------------------------------------------------------------
    # 5. CLEAN SCREENSHOTS (NO PII - to measure false positives)
    # -------------------------------------------------------------
    clean_scenarios = [
        {
            "name": "clean_analytics_01.png",
            "title": "Cluster Monitor — System Metrics & Prometheus Telemetry",
            "metrics": [
                "CPU Utilization: 24.8% (8 Cores online)",
                "Memory Footprint: 6.4 GB / 32 GB (Allocated)",
                "Cluster Throughput: 14,820 req/sec",
                "P99 Latency: 12.4 ms | P50 Latency: 3.1 ms",
                "HTTP 200 OK: 99.98% | Error Rate: 0.02%",
                "Storage Pool: 4.2 TB NVMe RAID-10 (Healthy)",
                "Active Nodes: worker-01, worker-02, worker-03"
            ]
        },
        {
            "name": "clean_code_editor_02.png",
            "title": "VS Code — DijkstraShortestPath.py",
            "metrics": [
                "def dijkstra(graph, start_vertex):",
                "    distances = {vertex: float('infinity') for vertex in graph}",
                "    distances[start_vertex] = 0",
                "    pq = [(0, start_vertex)]",
                "    while len(pq) > 0:",
                "        current_dist, current_v = heapq.heappop(pq)",
                "        if current_dist > distances[current_v]: continue",
                "    return distances"
            ]
        },
        {
            "name": "clean_docs_page_03.png",
            "title": "API Reference Documentation — Vector Index Service",
            "metrics": [
                "GET /v1/indexes/{index_name}/query",
                "Parameters: top_k (integer, default=10)",
                "Response: JSON array of vector similarity matches",
                "Algorithm: Hierarchical Navigable Small World (HNSW)",
                "Metric: Cosine distance normalized on unit sphere",
                "Status: Experimental release channel v2.4.0-rc1"
            ]
        },
        {
            "name": "clean_settings_04.png",
            "title": "Settings — Audio, Display & Network Hardware",
            "metrics": [
                "Display Resolution: 2560 x 1440 @ 144Hz (HDR On)",
                "Audio Output: Realtek High Definition Audio Device",
                "Refresh Rate: Adaptive Sync Enabled (G-Sync Compatible)",
                "Color Profile: DCI-P3 98% Hardware Calibrated",
                "Night Light Schedule: Sunset to Sunrise (3200K)",
                "Firmware Version: 14.2.0-build-889"
            ]
        }
    ]
    for sc in clean_scenarios:
        test_cases.append({
            "category": "CLEAN",
            "file": sc["name"],
            "title": sc["title"],
            "scenario": sc
        })

    return test_cases


# =====================================================================
# RENDERING ENGINE
# =====================================================================

def render_screenshot(case: Dict[str, Any]) -> Dict[str, Any]:
    width, height = 1000, 680
    img = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    
    cat = case["category"]
    sc = case["scenario"]
    draw_window_frame(draw, width, height, case["title"])
    
    ground_truth_items = []
    
    if cat == "EMAIL_CLIENT":
        # Sidebar
        draw.rectangle([0, 38, 220, height], fill=(241, 245, 249), outline=(226, 232, 240))
        draw.text((25, 60), "MAILBOXES", font=FONT_HEADING, fill=(71, 85, 105))
        folders = ["📥  Inbox (12)", "⭐  Starred", "📤  Sent Mail", "📝  Drafts (2)", "🗑️  Trash"]
        for idx, fld in enumerate(folders):
            draw.text((25, 95 + idx * 32), fld, font=FONT_BODY, fill=(51, 65, 85))
            
        # Email view card
        draw_card(draw, 240, 55, 975, 650)
        draw.text((265, 75), sc["subject"], font=FONT_TITLE, fill=(15, 23, 42))
        
        # From line
        draw.text((265, 115), "From:", font=FONT_BOLD, fill=(100, 116, 139))
        draw.text((320, 115), sc["sender"], font=FONT_BODY, fill=(15, 23, 42))
        
        # Date & status
        draw.text((820, 115), "Today, 10:24 AM", font=FONT_SMALL, fill=(148, 163, 184))
        draw.line([265, 145, 950, 145], fill=(226, 232, 240), width=1)
        
        # Body lines
        curr_y = 170
        for line in sc["lines"]:
            draw.text((265, curr_y), line, font=FONT_BODY, fill=(30, 41, 59))
            curr_y += 34
            
        # Record PII annotations
        ground_truth_items.append({"pii_type": "EMAIL", "text": sc["sender"]})
        ground_truth_items.append({"pii_type": "PHONE_IN", "text": sc["phone"]})
        ground_truth_items.append({"pii_type": "UPI_ID", "text": sc["upi"]})

    elif cat == "KYC_FORM":
        draw_card(draw, 60, 60, 940, 640)
        draw.text((95, 85), case["title"], font=FONT_TITLE, fill=(15, 23, 42))
        draw.text((95, 115), "Statutory Indian Identity Compliance — Form 60 / Aadhaar-PAN Linking", font=FONT_SMALL, fill=(100, 116, 139))
        draw.line([95, 140, 905, 140], fill=(226, 232, 240), width=1)
        
        # Form Fields
        fields = [
            ("Full Legal Name", sc["name_val"], False),
            ("Date of Birth", sc["dob"], False),
            ("Income Tax Permanent Account Number (PAN)", sc["pan"], "PAN"),
            ("Unique Identification Aadhaar Number (UIDAI)", f"{sc['aadhaar'][:4]} {sc['aadhaar'][4:8]} {sc['aadhaar'][8:]}", "AADHAAR"),
            ("Registered Mobile Number (OTP Verified)", sc["phone"], "PHONE_IN")
        ]
        
        curr_y = 165
        for label, val, pii_t in fields:
            draw.text((95, curr_y), label, font=FONT_BOLD, fill=(51, 65, 85))
            # Input Box
            draw.rounded_rectangle([95, curr_y + 24, 850, curr_y + 64], radius=6, fill=(248, 250, 252), outline=(203, 213, 225), width=1)
            draw.text((115, curr_y + 36), val, font=FONT_BODY, fill=(15, 23, 42))
            
            if pii_t:
                ground_truth_items.append({"pii_type": pii_t, "text": val})
            curr_y += 85
            
        # Submit Button
        draw.rounded_rectangle([95, 600, 280, 630], radius=6, fill=(37, 99, 235))
        draw.text((125, 606), "Submit & Verify KYC", font=FONT_BOLD, fill=(255, 255, 255))

    elif cat == "BANKING_DASHBOARD":
        # Header banner
        draw_card(draw, 50, 60, 950, 135, fill=(30, 58, 138), outline=(30, 58, 138))
        draw.text((80, 75), sc["title"], font=FONT_TITLE, fill=(255, 255, 255))
        draw.text((80, 105), sc["bank_name"], font=FONT_SMALL, fill=(191, 219, 254))
        
        # Cards grid
        # Card 1: Bank Account Details
        draw_card(draw, 50, 155, 480, 390)
        draw.text((75, 175), "PRIMARY SETTLEMENT ACCOUNT", font=FONT_HEADING, fill=(30, 41, 59))
        draw.text((75, 215), "Branch IFSC Code:", font=FONT_BOLD, fill=(100, 116, 139))
        draw.text((240, 215), sc["ifsc"], font=FONT_BODY, fill=(15, 23, 42))
        
        draw.text((75, 255), "Account Number:", font=FONT_BOLD, fill=(100, 116, 139))
        draw.text((240, 255), "50100294821038", font=FONT_BODY, fill=(15, 23, 42))
        
        draw.text((75, 295), "Linked Virtual VPA:", font=FONT_BOLD, fill=(100, 116, 139))
        draw.text((240, 295), sc["upi"], font=FONT_BODY, fill=(15, 23, 42))
        
        draw.text((75, 335), "Audit Notification Email:", font=FONT_BOLD, fill=(100, 116, 139))
        draw.text((255, 335), sc["email"], font=FONT_BODY, fill=(15, 23, 42))
        
        # Card 2: Commercial Credit Card
        draw_card(draw, 510, 155, 950, 390)
        draw.text((535, 175), "CORPORATE PURCHASING CARD", font=FONT_HEADING, fill=(30, 41, 59))
        draw.rounded_rectangle([535, 210, 925, 370], radius=10, fill=(15, 23, 42))
        draw.text((560, 230), "PLATINUM COMMERCIAL", font=FONT_SMALL, fill=(148, 163, 184))
        card_formatted = f"{sc['card'][:4]} {sc['card'][4:8]} {sc['card'][8:12]} {sc['card'][12:]}"
        draw.text((560, 275), card_formatted, font=FONT_TITLE, fill=(248, 250, 252))
        draw.text((560, 325), "VALID THRU: 08/29", font=FONT_SMALL, fill=(203, 213, 225))
        draw.text((750, 325), "VIJAYRAJ S", font=FONT_BOLD, fill=(255, 255, 255))
        
        # Bottom transaction ledger
        draw_card(draw, 50, 410, 950, 640)
        draw.text((75, 430), "RECENT REAL-TIME SETTLEMENTS", font=FONT_HEADING, fill=(30, 41, 59))
        draw.text((75, 470), "Txn ID #982103  |  RTGS Inward Clearing  |  IFSC: " + sc["ifsc"] + "  |  Status: SUCCESSFUL", font=FONT_BODY, fill=(51, 65, 85))
        draw.text((75, 510), "UPI Collect Req |  From: " + sc["upi"] + "  |  Amount: INR 45,000.00  |  Approved", font=FONT_BODY, fill=(51, 65, 85))
        draw.text((75, 550), "Statement Dispatch  |  Recipient: " + sc["email"] + "  |  Sent", font=FONT_BODY, fill=(51, 65, 85))
        
        ground_truth_items.append({"pii_type": "IFSC", "text": sc["ifsc"]})
        ground_truth_items.append({"pii_type": "UPI_ID", "text": sc["upi"]})
        ground_truth_items.append({"pii_type": "EMAIL", "text": sc["email"]})
        ground_truth_items.append({"pii_type": "CARD_NUMBER", "text": card_formatted})

    elif cat == "CHAT_SUPPORT":
        draw_card(draw, 100, 60, 900, 640)
        draw.text((130, 80), case["title"], font=FONT_HEADING, fill=(15, 23, 42))
        draw.line([100, 110, 900, 110], fill=(226, 232, 240), width=1)
        
        curr_y = 135
        for sender, msg in sc["messages"]:
            draw.text((130, curr_y), sender, font=FONT_BOLD, fill=(37, 99, 235))
            draw.text((130, curr_y + 22), msg, font=FONT_BODY, fill=(30, 41, 59))
            curr_y += 65
            
        for text_val, pii_t in sc["pii_items"]:
            ground_truth_items.append({"pii_type": pii_t, "text": text_val})

    elif cat == "CLEAN":
        draw_card(draw, 80, 70, 920, 630)
        draw.text((115, 95), case["title"], font=FONT_TITLE, fill=(15, 23, 42))
        draw.line([115, 130, 885, 130], fill=(226, 232, 240), width=1)
        
        curr_y = 160
        for item in sc["metrics"]:
            draw.text((115, curr_y), item, font=FONT_BODY, fill=(51, 65, 85))
            curr_y += 48
            
        # Clean case has NO ground truth PII items!

    out_path = os.path.join(OUTPUT_DIR, case["file"])
    img.save(out_path, "PNG")
    
    return {
        "file": case["file"],
        "category": cat,
        "is_clean": (cat == "CLEAN"),
        "ground_truth_count": len(ground_truth_items),
        "ground_truth_items": ground_truth_items
    }


def main():
    print(f"Generating 20 synthetic test screenshots into '{OUTPUT_DIR}'...")
    test_cases = create_synthetic_datasets()
    summary = []
    
    total_pii_items = 0
    for case in test_cases:
        res = render_screenshot(case)
        summary.append(res)
        total_pii_items += res["ground_truth_count"]
        status = "CLEAN (0 PII)" if res["is_clean"] else f"PII: {res['ground_truth_count']} entities"
        print(f"  [OK] {res['file']:<25} ({res['category']:<17}) -> {status}")
        
    gt_file = os.path.join(OUTPUT_DIR, "ground_truth.json")
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("=" * 60)
    print(f"Successfully generated {len(summary)} synthetic screenshots.")
    print(f"Total synthetic PII ground truth instances: {total_pii_items}")
    print(f"Clean (negative control) images: {sum(1 for s in summary if s['is_clean'])}")
    print(f"Ground truth catalog saved to: {gt_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
