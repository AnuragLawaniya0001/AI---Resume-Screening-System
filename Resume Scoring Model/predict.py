"""
Step 3: predict.py
===================
Load trained BERT model and predict score + breakdown from raw resume text.

Usage:
    python predict.py                          # demo with 3 sample resumes
    python predict.py --file path/to/resume.txt
    python predict.py --text "Your resume text here"

Output:
    {
      "total_score"      : 72.4,
      "grade"            : "Good",
      "breakdown": {
        "experience_score": 18.5,
        "education_score" : 20.0,
        "skills_score"    : 21.3,
        "structure_score" : 12.6
      },
      "suggestions": [...]
    }
"""

import os, json, argparse
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# ── Model definition (must match train_bert.py) ────────────────────────────

class BertResumeScorer(nn.Module):
    def __init__(self, bert_name, dropout=0.3, max_score=25.0):
        super().__init__()
        self.bert      = BertModel.from_pretrained(bert_name)
        hidden         = self.bert.config.hidden_size
        self.max_score = max_score
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        return self.regressor(pooled) * self.max_score

# ── Scorer class ────────────────────────────────────────────────────────────

class ResumeScorer:
    GRADE_THRESHOLDS = [(85,"Excellent"),(70,"Good"),(55,"Average"),(40,"Below Average"),(0,"Weak")]
    SUGGESTIONS = {
        "experience_score": {
            "low" : "Add more detail about your work experience — quantify achievements (e.g., '↑ performance by 30%').",
            "mid" : "Include specific metrics and outcomes for each role.",
            "high": "Experience section looks strong.",
        },
        "education_score": {
            "low" : "Mention your degree, institution, and graduation year clearly.",
            "mid" : "Consider adding relevant coursework or academic achievements.",
            "high": "Education section is well-presented.",
        },
        "skills_score": {
            "low" : "List more technical skills and add relevant certifications.",
            "mid" : "Group skills by category (Languages, Frameworks, Tools) for clarity.",
            "high": "Skills section is comprehensive.",
        },
        "structure_score": {
            "low" : "Add a professional summary, LinkedIn/GitHub links, and use bullet points.",
            "mid" : "Add GitHub or portfolio link to strengthen online presence.",
            "high": "Resume structure is professional.",
        },
    }

    def __init__(self, model_dir="model/bert_resume_scorer"):
        cfg_path = os.path.join(model_dir, "scorer_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"No scorer_config.json in '{model_dir}'.\n"
                "Run train_bert.py first to train and save the model."
            )

        with open(cfg_path) as f:
            self.cfg = json.load(f)

        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.targets   = self.cfg["targets"]
        self.max_score = self.cfg["max_score"]

        print(f"Loading tokenizer from {model_dir} ...")
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)

        print(f"Loading model from {model_dir} ...")
        self.model = BertResumeScorer(
            self.cfg["bert_name"],
            self.cfg["dropout"],
            self.cfg["max_score"],
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, "pytorch_model.bin"),
                       map_location=self.device)
        )
        self.model.eval()
        print("Model ready ✓")

    def _grade(self, score):
        for threshold, label in self.GRADE_THRESHOLDS:
            if score >= threshold:
                return label
        return "Weak"

    def _suggestions(self, breakdown):
        tips = []
        for key, val in breakdown.items():
            ratio = val / self.max_score
            level = "high" if ratio >= 0.72 else ("mid" if ratio >= 0.44 else "low")
            tip   = self.SUGGESTIONS[key][level]
            tips.append(f"[{key.replace('_score','').title()}] {tip}")
        return tips

    @torch.no_grad()
    def predict(self, resume_text: str) -> dict:
        """
        Args:
            resume_text : raw resume as plain string

        Returns:
            dict with total_score, grade, breakdown, suggestions
        """
        enc = self.tokenizer(
            resume_text,
            max_length=self.cfg["max_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids  = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        raw_scores = self.model(ids, mask).squeeze(0).cpu().tolist()   # 4 values

        breakdown = {
            name: round(score, 2)
            for name, score in zip(self.targets, raw_scores)
        }
        total = round(min(sum(raw_scores), 100.0), 2)

        return {
            "total_score" : total,
            "grade"       : self._grade(total),
            "breakdown"   : breakdown,
            "suggestions" : self._suggestions(breakdown),
        }

    def predict_batch(self, resume_texts: list) -> list:
        """Score multiple resumes at once."""
        return [self.predict(t) for t in resume_texts]


# ── CLI / Demo ───────────────────────────────────────────────────────────────

SAMPLE_RESUMES = [
    # ── Strong senior resume ──
    """Arjun Sharma
arjun.sharma@email.com | +91 98765 43210 | Bangalore
LinkedIn: linkedin.com/in/arjun-sharma | GitHub: github.com/arjun-ml

PROFESSIONAL SUMMARY
Senior Machine Learning Engineer with 8+ years of experience building production-grade
AI systems. Specialized in NLP and deep learning using Python, PyTorch, and TensorFlow.

WORK EXPERIENCE

Senior ML Engineer | Google India | 2020–Present
• Built a real-time NLP pipeline reducing customer query resolution time by 40%.
• Led a team of 6 engineers to deliver a recommendation engine serving 50M+ users.
• Reduced model inference cost by 35% through quantization and model distillation.

ML Engineer | Flipkart | 2017–2020
• Developed XGBoost-based fraud detection model saving $2M annually.
• Improved search ranking model's NDCG score from 0.72 to 0.89.

Software Engineer | Infosys | 2015–2017
• Built RESTful APIs using Django and PostgreSQL for e-commerce clients.

EDUCATION
Doctor of Philosophy – Computer Science
IISc Bangalore | 2015 | Thesis: Attention Mechanisms in Neural Machine Translation

TECHNICAL SKILLS
• Python, Java, C++, SQL, Bash
• PyTorch, TensorFlow, Scikit-learn, Keras, Hugging Face Transformers
• AWS, GCP, Docker, Kubernetes, Spark, Airflow
• NLP, Computer Vision, Reinforcement Learning, MLOps

CERTIFICATIONS
• Google Cloud Professional ML Engineer (2023)
• AWS Certified Solutions Architect – Associate (2022)

LANGUAGES
English, Hindi, Kannada
""",

    # ── Mid-level resume ──
    """Priya Verma
priya.verma@email.com | +91 87654 32109 | Pune

PROFESSIONAL SUMMARY
Backend developer with 3 years of experience in Python and Java.
Looking for challenging opportunities in software development.

WORK EXPERIENCE

Software Engineer | TCS | 2022–Present
• Developed REST APIs using Spring Boot for banking client.
• Worked with MySQL and Redis for data storage.

Junior Developer | Wipro | 2021–2022
• Maintained legacy Java codebase and wrote unit tests.

EDUCATION
Bachelor of Technology – Computer Science
VIT Vellore | 2021

TECHNICAL SKILLS
• Java, Python, SQL
• Spring Boot, Django, Hibernate
• MySQL, MongoDB, Git
• Agile, Scrum

LinkedIn: linkedin.com/in/priya-verma
""",

    # ── Weak fresher resume ──
    """Rahul Das
rahul@gmail.com | Delhi

EDUCATION
Diploma in Computer Science
Delhi Polytechnic | 2023

SKILLS
Python, HTML, CSS

PROJECTS
Made a to-do app in Python.
""",
]


def print_result(label, result):
    print(f"\n{'═'*55}")
    print(f"  {label}")
    print(f"{'═'*55}")
    print(f"  Total Score : {result['total_score']:5.1f} / 100   [{result['grade']}]")
    bar = "█" * int(result['total_score'] // 5) + "░" * (20 - int(result['total_score'] // 5))
    print(f"  [{bar}]")
    print(f"\n  Breakdown:")
    for key, val in result["breakdown"].items():
        name  = key.replace("_score","").replace("_"," ").title()
        b     = "█" * int(val // 1.25) + "░" * (20 - int(val // 1.25))
        print(f"    {name:12s}: {val:5.1f}/25  [{b}]")
    print(f"\n  Suggestions:")
    for tip in result["suggestions"]:
        print(f"    • {tip}")


def main():
    parser = argparse.ArgumentParser(description="BERT Resume Scorer")
    parser.add_argument("--file",  type=str, default=None, help="Path to resume .txt file")
    parser.add_argument("--text",  type=str, default=None, help="Resume text string")
    parser.add_argument("--model", type=str, default="model/bert_resume_scorer",
                        help="Path to saved model directory")
    args = parser.parse_args()

    scorer = ResumeScorer(model_dir=args.model)

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
        result = scorer.predict(text)
        print_result(f"File: {args.file}", result)

    elif args.text:
        result = scorer.predict(args.text)
        print_result("Custom Resume", result)

    else:
        # run all 3 demo resumes
        labels = ["Senior ML Engineer (Strong)", "Mid-level Backend Dev", "Fresher (Weak)"]
        for label, resume in zip(labels, SAMPLE_RESUMES):
            result = scorer.predict(resume)
            print_result(label, result)


if __name__ == "__main__":
    main()
