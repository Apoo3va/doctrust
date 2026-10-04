"""
testset.py
A small hand-written test set covering all three source types (PDF, wiki, CSV),
used to evaluate DocTrust's answer quality with RAGAS. Expanded in Stage 10 to
cover the larger, messier dataset -- including a deliberately conflicting fact
across two sources, used to check whether the pipeline surfaces the conflict
rather than silently picking one answer.
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
    # --- Stage 10: new, messier multi-source dataset ---
    {
        "question": "What is the PF contribution rate?",
        "ground_truth": "Employees contribute 12% of basic salary to PF, matched by the company.",
        "source_type": "wiki+csv",
    },
    {
        "question": "How do I reset my password using self-service?",
        "ground_truth": (
            "Use the self-service password reset tool linked on the company intranet "
            "homepage. For urgent lockouts outside business hours, call the 24x7 IT helpline."
        ),
        "source_type": "wiki",
    },
    {
        "question": "What happens if an employee receives a low performance rating twice?",
        "ground_truth": (
            "They are placed on a 60-day Performance Improvement Plan (PIP), co-created "
            "with HR and the reporting manager, with a check-in every 2 weeks."
        ),
        "source_type": "pdf",
    },
    {
        # Intentionally conflicting fact across two sources (PDF says INR 1800,
        # CSV FAQ says INR 1500). Ground truth documents both figures so RAGAS
        # scores reflect whether the system surfaces the conflict honestly
        # rather than confidently stating a single number.
        "question": "What is the daily travel allowance for domestic trips?",
        "ground_truth": (
            "Sources disagree: the Travel and Reimbursement Policy PDF states INR 1800 "
            "per day for meals and incidentals, while the FAQ records list INR 1500. "
            "A trustworthy answer should surface both figures and the conflict rather "
            "than confidently stating only one."
        ),
        "source_type": "pdf+csv (conflict)",
    },
]