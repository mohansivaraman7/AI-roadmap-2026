"""Writes corpus.txt: a few thousand varied sentences about a made-up company (Acme).
Replace corpus.txt with your own text (any .txt) to train on your domain."""
import random
random.seed(7)
topics = [
 ("annual leave", ["Full-time employees receive {n} days of annual leave each year.", "Annual leave should be requested at least {m} days in advance.", "Up to {k} unused days of annual leave can be carried forward."], dict(n=[20], m=[7], k=[5])),
 ("sick leave", ["Employees get {n} days of sick leave per year.", "A medical certificate is needed after {m} days of absence."], dict(n=[10], m=[3])),
 ("remote work", ["Employees may work from home {n} days per week with manager approval.", "Remote work requests are approved by the line manager."], dict(n=[2])),
 ("pay", ["Salaries are paid on the last working day of every month.", "Payslips are available in the HR portal after payment."], {}),
 ("hours", ["Working hours are 9 AM to 6 PM, Monday to Friday.", "Everyone gets a one hour lunch break."], {}),
 ("reviews", ["Performance reviews happen once every year in March.", "Managers share written feedback before the review meeting."], {}),
 ("it", ["Reset your password at the self service portal.", "Call the IT helpdesk on extension 4400 for laptop problems.", "Connect to the VPN before opening internal tools."], {}),
 ("travel", ["Business travel must be approved by your manager before booking.", "Submit travel expenses within {n} days of returning."], dict(n=[14, 30])),
]
openers = ["", "Please note: ", "Reminder: ", "Policy: ", "For all staff: "]
lines = []
for _ in range(1500):
    name, templates, params = random.choice(topics)
    t = random.choice(templates)
    vals = {k: random.choice(v) for k, v in params.items()}
    # keep numbers consistent per topic: use first value so facts stay true
    vals = {k: v[0] for k, v in params.items()}
    lines.append(random.choice(openers) + t.format(**vals))
open("corpus.txt", "w").write("\n".join(lines) + "\n")
print(len(lines), "lines")
