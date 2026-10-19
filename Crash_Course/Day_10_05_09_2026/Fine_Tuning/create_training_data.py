"""
create_training_data.py -- builds a tiny Q&A dataset from the SAME
company_policy.pdf content you've used throughout this course, but
shaped for FINE-TUNING instead of RETRIEVAL.

This is the key conceptual difference to point out to students:
  - For RAG (create_db_with_pdf.py): you split the PDF into CHUNKS,
    embedded them, and searched them at query time. The model itself
    never changes.
  - For LoRA/QLoRA (this file + lora_finetune.py): you turn the SAME
    policy content into QUESTION -> ANSWER pairs, and train the model's
    own weights on them. After training, the model answers from what
    it "learned", not from anything retrieved at query time.

RUN:
    python create_training_data.py
Produces: policy_training_data.jsonl
"""

import json

# Same facts as company_policy.pdf, reshaped as instruction/response pairs
TRAINING_EXAMPLES = [
    {"instruction": "How many days of annual leave do employees get?",
     "response": "All full-time employees receive 20 days of annual leave per calendar year."},
    {"instruction": "How far in advance should annual leave be requested?",
     "response": "Annual leave should normally be requested at least 7 days in advance."},
    {"instruction": "Can unused annual leave be carried forward?",
     "response": "Unused annual leave can be carried forward up to 5 days into the following year."},
    {"instruction": "How many sick leave days do employees get per year?",
     "response": "Employees receive 10 days of sick leave per year."},
    {"instruction": "When is a medical certificate required for sick leave?",
     "response": "If an employee is absent for more than 3 consecutive working days, a medical certificate may be required."},
    {"instruction": "How many days per week can employees work from home?",
     "response": "Employees can work from home 2 days per week, with manager approval."},
    {"instruction": "What are the normal working hours?",
     "response": "Normal working hours are 9:00 AM to 6:00 PM, Monday to Friday, with a 1-hour lunch break."},
    {"instruction": "When are employees paid?",
     "response": "Employees are paid on the last working day of each month."},
    {"instruction": "How often do employees receive a performance review?",
     "response": "Employees receive a formal performance review once every year."},
]

with open("policy_training_data.jsonl", "w") as f:
    for example in TRAINING_EXAMPLES:
        # Standard instruction-tuning format most fine-tuning scripts expect
        text = f"### Question:\n{example['instruction']}\n\n### Answer:\n{example['response']}"
        f.write(json.dumps({"text": text}) + "\n")

print(f"Wrote {len(TRAINING_EXAMPLES)} examples to policy_training_data.jsonl")
