"""
generate_stage10_data.py
Stage 10: generates a larger, messier dataset for DocTrust to stress-test
retrieval, routing, and guardrails. Adds 3 new PDFs, 3 new wiki pages, and
~15 new CSV rows -- including a couple of intentionally conflicting facts
to test whether the validator agent catches contradictions.

Run once from the project root:
    python generate_stage10_data.py
"""

import csv
from pathlib import Path
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "data" / "pdf"
WIKI_DIR = ROOT / "data" / "wiki"
RECORDS_DIR = ROOT / "data" / "records"

PDF_DIR.mkdir(parents=True, exist_ok=True)
WIKI_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR.mkdir(parents=True, exist_ok=True)


def make_pdf(filename: str, title: str, sections: list[tuple[str, str]]):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, title)
    pdf.ln(4)
    for heading, body in sections:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, heading)
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, body)
        pdf.ln(3)
    pdf.output(str(PDF_DIR / filename))
    print(f"Created {filename}")


# ---------------------------------------------------------------------------
# PDF 1: IT Security Policy (messy formatting, one typo on purpose)
# ---------------------------------------------------------------------------
make_pdf(
    "it_security_policy.pdf",
    "IT Security Policy",
    [
        (
            "1. Passowrd Requirements",
            "All employees must use a passowrd that is at least 10 characters long, "
            "including one uppercase letter, one number, and one special character. "
            "Passwords must be changed every 90 days. Reusing the last 5 passwords is "
            "not allowed.",
        ),
        (
            "2 - Device Encryption",
            "- All laptops issued by the company must have full-disk encryption enabled.\n"
            "- Personal devices used for work (BYOD) must install the MobileIron MDM agent.\n"
            "* USB storage devices are restricted unless approved by IT Security.",
        ),
        (
            "Incident Reporting",
            "Any suspected security incident (lost device, phishing email, suspicious login) "
            "must be reported to security@company.com within 2 hours of discovery. IT "
            "Security will respond within 4 business hours for high severity incidents.",
        ),
        (
            "Acceptable Use",
            "Company systems are for business use. Limited personal use is tolerated as "
            "long as it does not interfere with work, consume excessive bandwidth, or "
            "violate any other company policy.",
        ),
    ],
)

# ---------------------------------------------------------------------------
# PDF 2: Travel & Reimbursement Policy
# (deliberately uses a different per-diem figure than the CSV FAQ, to test
#  whether the system flags the conflict instead of blending the numbers)
# ---------------------------------------------------------------------------
make_pdf(
    "travel_reimbursement_policy.pdf",
    "Travel and Reimbursement Policy",
    [
        (
            "Domestic Travel",
            "Employees traveling domestically for business purposes are entitled to a "
            "daily allowance of INR 1800 for meals and incidentals. Hotel bookings above "
            "INR 6000 per night require manager pre-approval.",
        ),
        (
            "International Travel",
            "International travel requires VP-level approval at least 2 weeks in advance. "
            "The standard daily allowance for international trips is USD 60, covering "
            "meals and local transport. Flights must be booked through the approved "
            "corporate travel portal.",
        ),
        (
            "Expense Submission",
            "All expenses must be submitted within 15 days of travel completion via the "
            "Expense Portal. Original receipts are required for any single expense over "
            "INR 500. Expenses submitted late may be rejected at the discretion of Finance.",
        ),
        (
            "Approval Limits",
            "Managers can approve expenses up to INR 25000 per trip. Anything above this "
            "amount requires Finance Director approval regardless of department.",
        ),
    ],
)

# ---------------------------------------------------------------------------
# PDF 3: Performance Review Policy
# ---------------------------------------------------------------------------
make_pdf(
    "performance_review_policy.pdf",
    "Performance Review Policy",
    [
        (
            "Review Cycle",
            "Performance reviews are conducted twice a year: a mid-year check-in in "
            "July and an annual review in January. New employees receive an additional "
            "90-day probation review.",
        ),
        (
            "Rating Scale",
            "Employees are rated on a 5-point scale: 1 = Needs Improvement, "
            "2 = Developing, 3 = Meets Expectations, 4 = Exceeds Expectations, "
            "5 = Outstanding. Ratings feed into the annual compensation review.",
        ),
        (
            "Performance Improvement Plan (PIP)",
            "Employees rated 'Needs Improvement' for two consecutive cycles are placed "
            "on a 60-day Performance Improvement Plan. The PIP is co-created with HR and "
            "the reporting manager, with clear measurable goals and a check-in every 2 weeks.",
        ),
        (
            "Appeals Process",
            "Employees who disagree with their rating can file an appeal with HR within "
            "10 business days of receiving the review. A second-level manager will review "
            "the case and respond within 15 business days.",
        ),
    ],
)

# ---------------------------------------------------------------------------
# Wiki page 1: Benefits Overview
# ---------------------------------------------------------------------------
(WIKI_DIR / "benefits_overview.md").write_text(
    """# Benefits Overview

## Health Insurance
All full-time employees are covered under the group health insurance plan from
day one of employment. Coverage extends to spouse, children, and dependent
parents. The sum insured is INR 5,00,000 per family per year.

## Provident Fund (PF)
Employees contribute 12% of basic salary to PF, matched by the company. PF
withdrawals are allowed after 2 months of continuous unemployment or for
specific reasons like home purchase or medical emergencies, per government rules.

## Gratuity
Employees who complete 5 or more years of continuous service are eligible for
gratuity as per the Payment of Gratuity Act.

## Wellness Allowance
Every employee gets an annual wellness allowance of INR 10,000, reimbursable
against gym memberships, fitness classes, or mental health counseling sessions.
""",
    encoding="utf-8",
)
print("Created benefits_overview.md")

# ---------------------------------------------------------------------------
# Wiki page 2: Remote Work FAQ
# (intentionally slightly conflicts with onboarding.md's "2 days WFH/week")
# ---------------------------------------------------------------------------
(WIKI_DIR / "remote_work_faq.md").write_text(
    """# Remote Work FAQ

**Q: How many days can I work from home?**
Most teams are fine with employees working from home up to 3 days a week,
as long as your manager is okay with it and you're still around for key
meetings.

**Q: Do I need to log my WFH days anywhere?**
Yes, mark your WFH days on the team calendar so people know when you're
remote.

**Q: Can interns work remotely?**
Interns are generally expected to be onsite during their internship period
unless there's a specific reason approved by their mentor.

**Q: What if I need to work from a different city temporarily?**
Let your manager and HR know at least a week in advance. Working from a
different country requires additional approval due to tax and compliance
reasons.
""",
    encoding="utf-8",
)
print("Created remote_work_faq.md")

# ---------------------------------------------------------------------------
# Wiki page 3: IT Support Guide
# ---------------------------------------------------------------------------
(WIKI_DIR / "it_support_guide.md").write_text(
    """# IT Support Guide

## Raising a Ticket
Raise an IT ticket through the internal helpdesk portal. Choose the category
that best matches your issue (Hardware, Software, Network, Access) to route
it to the right team faster.

## VPN Setup
1. Download the company VPN client from the software portal.
2. Log in with your company email and password.
3. Select the "Corporate" profile and connect.
4. Contact IT if you see a certificate error -- this usually means your
   device needs a security update.

## Password Reset
If you're locked out, use the self-service password reset tool linked on the
company intranet homepage. For urgent lockouts outside business hours, call
the 24x7 IT helpline.

## Hardware Requests
New hardware requests (laptop, monitor, headset) go through your manager for
budget approval, then to IT for fulfillment. Standard turnaround is 5
business days.
""",
    encoding="utf-8",
)
print("Created it_support_guide.md")

# ---------------------------------------------------------------------------
# CSV: append ~15 new FAQ rows, including one near-duplicate with a
# different answer than the PDF policy (conflict test)
# ---------------------------------------------------------------------------
csv_path = RECORDS_DIR / "faq_fixed.csv"

new_rows = [
    ["question", "answer", "department", "category"],
    ["What is the daily travel allowance for domestic trips?",
     "The daily travel allowance for domestic trips is INR 1500 for meals and incidentals.",
     "Finance", "travel"],
    ["How do I raise an IT support ticket?",
     "Raise a ticket through the internal helpdesk portal and choose the right category.",
     "IT", "it_support"],
    ["How often are performance reviews conducted?",
     "Performance reviews happen twice a year, in July and January.",
     "HR", "performance"],
    ["What is the PF contribution rate?",
     "Employees contribute 12% of basic salary to PF, matched by the company.",
     "HR", "benefits"],
    ["How long is the wellness allowance?",
     "The annual wellness allowance is INR 10,000 per employee.",
     "HR", "benefits"],
    ["What rating scale is used for performance reviews?",
     "A 5 point scale from Needs Improvement to Outstanding is used.",
     "HR", "performance"],
    ["How do I reset my password?",
     "Use the self service password reset tool on the company intranet homepage.",
     "IT", "it_support"],
    ["What happens if I get a low performance rating twice?",
     "You are placed on a 60 day Performance Improvement Plan with HR and your manager.",
     "HR", "performance"],
    ["Is health insurance available from day one?",
     "Yes, health insurance coverage starts from day one of employment.",
     "HR", "benefits"],
    ["How many days in advance should I request international travel approval?",
     "International travel requires VP level approval at least 2 weeks in advance.",
     "Finance", "travel"],
    ["What is the approval limit for managers on travel expenses?",
     "Managers can approve travel expenses up to INR 25000 per trip.",
     "Finance", "travel"],
    ["How long does a new hardware request take?",
     "Standard turnaround for new hardware requests is 5 business days.",
     "IT", "it_support"],
    ["Can I appeal my performance rating?",
     "Yes, you can file an appeal with HR within 10 business days of receiving your review.",
     "HR", "performance"],
    ["What is required for devices used for work?",
     "Company laptops must have full disk encryption enabled, and personal devices need the MDM agent installed.",
     "IT", "it_security"],
    ["Who do I report a lost laptop to?",
     "Report it to security@company.com within 2 hours of discovering it is lost.",
     "IT", "it_security"],
]

with open(csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # skip header row (already exists in file), write only data rows
    for row in new_rows[1:]:
        writer.writerow(row)

print(f"Appended {len(new_rows) - 1} rows to faq_fixed.csv")
print("\nStage 10 data generation complete.")