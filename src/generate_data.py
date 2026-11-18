import random
import pandas as pd

random.seed(42)

templates = {
    "Billing": {
        "High": [
            "I was double charged ${amt} on my credit card today. Refund immediately!",
            "My account was billed ${amt} after I already canceled my subscription.",
            "Unauthorized charge of ${amt} appeared on my invoice #{inv}."
        ],
        "Medium": [
            "Can I get a copy of my tax invoice #{inv} for last month?",
            "My promo code did not apply to my recent renewal payment.",
            "How do I update my company VAT number for future billing cycles?"
        ],
        "Low": [
            "Where can I view my past payment receipts in the dashboard?",
            "Does your annual plan offer a discount compared to monthly billing?",
            "Just checking what date my next automatic renewal is scheduled for."
        ]
    },
    "Technical Support": {
        "High": [
            "Production API is returning 500 Internal Server Error for all requests!",
            "Our entire team is locked out because the app crashes right after login.",
            "Data export job has been stuck at 0% for 6 hours and blocking our release."
        ],
        "Medium": [
            "Webhook notifications are delayed by about 15 minutes today.",
            "CSV upload fails whenever the file contains special characters.",
            "Dark mode layout overlaps text on the analytics reporting page."
        ],
        "Low": [
            "Is there a keyboard shortcut to collapse the left navigation sidebar?",
            "Can you share documentation on setting up custom webhook headers?",
            "Minor typo found in the tooltip under the export settings menu."
        ]
    },
    "Account Access": {
        "High": [
            "I suspect my account was compromised. Please freeze access immediately!",
            "Admin user left the company and we cannot access the primary workspace.",
            "Two-factor authentication codes are not arriving and I cannot sign in."
        ],
        "Medium": [
            "Password reset email link says expired as soon as I click it.",
            "Need help transferring workspace ownership to another team member.",
            "SSO login with Google throws a permissions mismatch error."
        ],
        "Low": [
            "How do I change the display name shown on my user profile?",
            "Where can I add a backup email address to my account settings?",
            "Can I link my GitHub account to my existing email login?"
        ]
    },
    "Feature Request": {
        "High": [
            "We need SAML SSO support before we can roll this out to 500 employees.",
            "Audit logs for user role changes are required for our compliance audit.",
            "Please add role-based access control for read-only contractors."
        ],
        "Medium": [
            "It would be very helpful to schedule automated weekly PDF reports.",
            "Can you add Slack integration for real-time ticket alerts?",
            "Support for bulk editing tags across multiple records at once."
        ],
        "Low": [
            "Would love a compact table view option on smaller laptop screens.",
            "Adding custom color themes for workspace branding would be nice.",
            "Could you allow sorting saved filters alphabetically?"
        ]
    }
}

rows = []
for i in range(600):
    category = random.choice(list(templates.keys()))
    priority = random.choice(["Low", "Medium", "High"])
    template = random.choice(templates[category][priority])
    
    text = template.format(
        amt=random.randint(29, 499),
        inv=random.randint(10000, 99999)
    ) + f" (ref-{i+1000})"
    
    rows.append({"text": text, "category": category, "priority": priority})

df = pd.DataFrame(rows)
df.to_csv("data/raw/raw_tickets.csv", index=False)
print("Saved 600 raw tickets to data/raw/raw_tickets.csv")
