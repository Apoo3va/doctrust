"""
testset.py
A small hand-written test set covering all three source types (PDF, wiki, CSV),
used to evaluate DocTrust's answer quality with RAGAS.
"""

TEST_SET = [
    {
        "question": "What is the HR leave policy?",
        "ground_truth": (
            "Employees accrue 1.5 days of paid leave per month, capped at 18 days per year. "
            "Planned leave requires 3 business days advance notice through the HR portal. "
            "Sick leave does not require advance notice but must be reported the same day. "
            "Up to 5 unused leave days can be carried forward to the next year."
        ),
        "source_type": "wiki+csv",
    },
    {
        "question": "How do I reset my VPN password?",
        "ground_truth": (
            "Go to the IT portal and click 'Reset Credentials'. A new password is emailed "
            "within 15 minutes."
        ),
        "source_type": "csv",
    },
    {
        "question": "What is the company's data security policy?",
        "ground_truth": (
            "Employees must not store confidential company data on personal devices. "
            "Company laptops are encrypted by default. Any suspected data breach must be "
            "reported to the Security team within 24 hours of discovery."
        ),
        "source_type": "pdf",
    },
    {
        "question": "What is the reimbursement limit for client dinners?",
        "ground_truth": "Client dinner expenses are capped at INR 3000 per person without prior approval.",
        "source_type": "csv",
    },
    {
        "question": "What is the company's remote work eligibility policy?",
        "ground_truth": (
            "Remote work eligibility depends on role type and department needs. Engineering "
            "and design roles are generally eligible for hybrid arrangements. Client-facing "
            "roles require manager approval on a case-by-case basis."
        ),
        "source_type": "pdf",
    },
    {
        "question": "Is there a probation period for new hires?",
        "ground_truth": "New hires undergo a 6-month probation period before confirmation.",
        "source_type": "csv",
    },
]