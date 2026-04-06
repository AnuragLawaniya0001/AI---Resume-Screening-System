"""
Step 1: generate_dataset.py
============================
Simulates 3000 realistic resume texts with 4 sub-scores + total score.

Sub-scores (each 0-25):
  - experience_score  : years, companies, promotions, gaps
  - education_score   : degree level, relevant field, institution
  - skills_score      : tech skills count, certifications
  - structure_score   : summary, bullets, quantified achievements, online presence

Total score = sum of 4 sub-scores (0-100)

Output: data/resume_dataset.csv
  Columns: resume_text, experience_score, education_score,
           skills_score, structure_score, total_score
"""

import random
import re
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# ── Vocabulary pools ──────────────────────────────────────────────────────────

TECH_SKILLS_POOL = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "Kotlin", "Swift", "R", "Scala", "MATLAB",
    "React", "Angular", "Vue.js", "Django", "Flask", "FastAPI", "Spring Boot",
    "Node.js", "Express.js", "Next.js", "Laravel",
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
    "Oracle DB", "SQLite", "DynamoDB",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Jenkins",
    "GitHub Actions", "CI/CD", "Ansible", "Helm",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Data Science", "Pandas", "NumPy", "Scikit-learn", "TensorFlow",
    "PyTorch", "Keras", "XGBoost", "Spark", "Hadoop", "Tableau", "Power BI",
    "REST API", "GraphQL", "Microservices", "Agile", "Scrum", "Jira",
    "Linux", "Bash", "Git", "HTML", "CSS", "Sass", "Webpack", "Figma",
]

CERT_POOL = [
    "AWS Certified Solutions Architect",
    "Google Cloud Professional Data Engineer",
    "Microsoft Azure Developer Associate",
    "Certified Kubernetes Administrator (CKA)",
    "PMP – Project Management Professional",
    "Certified Scrum Master (CSM)",
    "TensorFlow Developer Certificate",
    "Oracle Certified Professional Java Developer",
    "CompTIA Security+",
    "Cisco CCNA",
]

EDU_LEVELS = {
    1: ("High School Diploma",      "Springfield High School"),
    2: ("Diploma in Computer Science", "Delhi Polytechnic Institute"),
    3: ("Bachelor of Technology",   ["IIT Delhi","NIT Trichy","VIT Vellore",
                                     "BITS Pilani","DTU Delhi","Amity University"]),
    4: ("Master of Technology",     ["IIT Bombay","IIT Madras","NIT Surathkal",
                                     "IIIT Hyderabad","IISc Bangalore"]),
    5: ("Doctor of Philosophy",     ["IIT Kharagpur","IISc Bangalore","IIT Kanpur"]),
}

COMPANIES = [
    "Infosys","TCS","Wipro","HCL Technologies","Tech Mahindra",
    "Accenture","Capgemini","IBM","Cognizant","Persistent Systems",
    "Flipkart","Paytm","Swiggy","Zomato","Razorpay","CRED","Ola","PhonePe",
    "Google","Microsoft","Amazon","Adobe","Oracle","SAP","Cisco","Intel",
    "JP Morgan","Goldman Sachs","Morgan Stanley","Deloitte","PwC",
]

ROLES = [
    "Software Engineer","Senior Software Engineer","Lead Engineer",
    "Principal Engineer","Software Developer","Backend Developer",
    "Frontend Developer","Full Stack Developer","Data Scientist",
    "Machine Learning Engineer","Data Analyst","DevOps Engineer",
    "Cloud Engineer","Site Reliability Engineer","QA Engineer",
    "Product Manager","Business Analyst","System Architect",
]

ACHIEVEMENTS = [
    "Reduced API response time by {n}% through caching and query optimization.",
    "Increased test coverage from {a}% to {b}%, reducing production bugs by {n}%.",
    "Led a team of {n} engineers to deliver the project {m} weeks ahead of schedule.",
    "Saved ${amt}K annually by automating manual reporting workflows.",
    "Scaled the platform to handle {n}M+ daily active users.",
    "Improved model accuracy from {a}% to {b}% using ensemble techniques.",
    "Reduced deployment time by {n}% by implementing CI/CD pipelines.",
    "Built a recommendation engine that increased user engagement by {n}%.",
    "Migrated {n} legacy microservices to Kubernetes, reducing infra cost by {m}%.",
    "Processed {n}GB of data daily using optimized Spark pipelines.",
]

OBJECTIVES = [
    "Passionate software engineer with {y}+ years of experience building scalable systems.",
    "Results-driven developer with expertise in {skill} and a track record of delivering impactful solutions.",
    "Experienced {role} seeking to leverage {y} years of expertise in a challenging environment.",
    "Detail-oriented engineer with strong foundation in {skill}, committed to code quality and performance.",
    "Dynamic professional with {y}+ years in software development, specializing in {skill}.",
]

LANGUAGES = ["English", "Hindi", "Tamil", "Telugu", "Marathi", "Bengali", "Gujarati"]

# ── Builder functions ─────────────────────────────────────────────────────────

def rand_achievement():
    tmpl = random.choice(ACHIEVEMENTS)
    return tmpl.format(
        n=random.randint(15, 80),
        a=random.randint(40, 70),
        b=random.randint(75, 95),
        m=random.randint(2, 12),
        amt=random.randint(50, 500),
    )

def build_experience_section(yrs, n_jobs, promoted):
    """Build realistic experience section; return (text, exp_score)."""
    if yrs == 0:
        return "No prior work experience.", 0.0

    lines = ["\nWORK EXPERIENCE"]
    remaining = yrs * 12  # months
    jobs_so_far = 0
    has_quant = False

    # distribute months across jobs
    job_durations = []
    for i in range(n_jobs):
        if i == n_jobs - 1:
            job_durations.append(remaining)
        else:
            dur = max(3, int(remaining / (n_jobs - i) + random.randint(-4, 4)))
            dur = min(dur, remaining - 3 * (n_jobs - i - 1))
            job_durations.append(dur)
            remaining -= dur

    # build from most recent to oldest
    end_year = 2025
    for i, dur in enumerate(reversed(job_durations)):
        months = dur
        start_year = end_year - (months // 12)
        company = random.choice(COMPANIES)
        role = random.choice(ROLES)
        if promoted and i == 0 and yrs >= 3:
            role = "Senior " + role.replace("Senior ", "")

        lines.append(f"\n{role} | {company} | {start_year}–{'Present' if i==0 else end_year}")
        lines.append(f"Duration: {months // 12} yr{'s' if months//12 != 1 else ''}"
                     + (f" {months % 12} mo" if months % 12 else ""))

        # bullets
        n_bullets = random.randint(2, 5)
        for _ in range(n_bullets):
            if random.random() < 0.55:
                lines.append(f"• {rand_achievement()}")
                has_quant = True
            else:
                lines.append(f"• Worked on {random.choice(TECH_SKILLS_POOL)} related tasks "
                             f"and collaborated with cross-functional teams.")
        end_year = start_year
        jobs_so_far += 1

    # score
    exp_score = min(25.0,
        yrs * 1.5
        + n_jobs * 0.5
        + (3.0 if promoted else 0.0)
        + (2.0 if has_quant else 0.0)
    )
    return "\n".join(lines), round(exp_score, 2)


def build_education_section(edu_level):
    """Return (text, edu_score)."""
    degree, inst_raw = EDU_LEVELS[edu_level]
    institution = random.choice(inst_raw) if isinstance(inst_raw, list) else inst_raw
    grad_year = random.randint(2005, 2023)
    lines = ["\nEDUCATION",
             f"{degree}",
             f"{institution} | Graduated {grad_year}"]

    # extra lines for higher edu
    if edu_level >= 3:
        lines.append(f"Major: {random.choice(['Computer Science','Information Technology','Electronics','Data Science','AI & ML'])}")
    if edu_level >= 4:
        lines.append(f"Thesis: {random.choice(['Deep Learning for NLP','Distributed Systems Optimization','Computer Vision Applications'])}")

    # score: 5*level but cap at 25
    edu_score = min(25.0, edu_level * 5.0)
    return "\n".join(lines), edu_score


def build_skills_section(n_skills, n_certs):
    """Return (text, skills_score)."""
    chosen = random.sample(TECH_SKILLS_POOL, min(n_skills, len(TECH_SKILLS_POOL)))
    lines = ["\nTECHNICAL SKILLS", "• " + ", ".join(chosen)]

    cert_lines = []
    if n_certs > 0:
        chosen_certs = random.sample(CERT_POOL, min(n_certs, len(CERT_POOL)))
        lines.append("\nCERTIFICATIONS")
        for c in chosen_certs:
            yr = random.randint(2018, 2024)
            lines.append(f"• {c} ({yr})")
        cert_lines = chosen_certs

    skills_score = min(25.0, n_skills * 0.8 + n_certs * 2.5)
    return "\n".join(lines), round(skills_score, 2), chosen


def build_structure_section(has_summary, has_linkedin, has_github,
                             has_portfolio, yrs, first_skill):
    """Return (summary_text, links_text, struct_score)."""
    summary = ""
    if has_summary:
        tmpl = random.choice(OBJECTIVES)
        summary = "PROFESSIONAL SUMMARY\n" + tmpl.format(
            y=max(yrs, 1),
            skill=first_skill,
            role=random.choice(ROLES),
        )

    links = []
    if has_linkedin:  links.append("LinkedIn: linkedin.com/in/candidate")
    if has_github:    links.append("GitHub: github.com/candidate")
    if has_portfolio: links.append("Portfolio: candidate.dev")
    links_text = "\n".join(links)

    struct_score = min(25.0,
        (5.0 if has_summary else 0.0)
        + (5.0 if has_linkedin else 0.0)
        + (7.0 if has_github else 0.0)
        + (4.0 if has_portfolio else 0.0)
        + random.uniform(0, 4.0)   # bonus for overall neatness
    )
    return summary, links_text, round(struct_score, 2)


def build_resume(params):
    """Assemble one full resume text + all sub-scores."""
    yrs        = params["yrs"]
    n_jobs     = params["n_jobs"]
    edu_level  = params["edu_level"]
    n_skills   = params["n_skills"]
    n_certs    = params["n_certs"]
    promoted   = params["promoted"]
    has_summary= params["has_summary"]
    has_linkedin=params["has_linkedin"]
    has_github = params["has_github"]
    has_portfolio=params["has_portfolio"]
    has_langs  = params["has_langs"]

    # name / header
    first = random.choice(["Aarav","Rohit","Priya","Ananya","Kunal","Neha",
                            "Arjun","Pooja","Vikram","Divya","Rahul","Sneha",
                            "Amit","Shreya","Karan","Meera","Nikhil","Tanvi"])
    last  = random.choice(["Sharma","Verma","Singh","Gupta","Patel","Kumar",
                            "Joshi","Mishra","Agarwal","Rao","Nair","Iyer",
                            "Reddy","Pillai","Mehta","Shah","Das","Bose"])
    name  = f"{first} {last}"
    email = f"{first.lower()}.{last.lower()}@email.com"
    phone = f"+91 {random.randint(70000,99999)}{random.randint(10000,99999)}"
    city  = random.choice(["Bangalore","Mumbai","Delhi","Hyderabad","Pune",
                            "Chennai","Kolkata","Noida","Gurgaon","Ahmedabad"])

    # sections
    exp_text,  exp_sc   = build_experience_section(yrs, n_jobs, promoted)
    edu_text,  edu_sc   = build_education_section(edu_level)
    skill_text,skill_sc, chosen_skills = build_skills_section(n_skills, n_certs)
    first_skill = chosen_skills[0] if chosen_skills else "software development"
    summ_text, links_text, struct_sc = build_structure_section(
        has_summary, has_linkedin, has_github, has_portfolio, yrs, first_skill)

    # languages
    lang_text = ""
    if has_langs:
        chosen_langs = random.sample(LANGUAGES, random.randint(1, 3))
        lang_text = "\nLANGUAGES\n• " + ", ".join(chosen_langs)

    # assemble
    header = f"{name}\n{email} | {phone} | {city}"
    parts  = [header]
    if summ_text:  parts.append(summ_text)
    parts += [exp_text, edu_text, skill_text]
    if lang_text:  parts.append(lang_text)
    if links_text: parts.append("\nONLINE PRESENCE\n" + links_text)

    resume_text = "\n".join(parts)

    total = exp_sc + edu_sc + skill_sc + struct_sc

    return {
        "resume_text":       resume_text,
        "experience_score":  exp_sc,
        "education_score":   edu_sc,
        "skills_score":      skill_sc,
        "structure_score":   struct_sc,
        "total_score":       round(min(total, 100.0), 2),
    }


# ── Main simulation ───────────────────────────────────────────────────────────

def simulate_dataset(n=3000):
    records = []
    for _ in range(n):
        yrs       = random.randint(0, 18)
        n_jobs    = random.randint(1, max(1, yrs // 2 + 1))
        edu_level = random.choices([1,2,3,4,5], weights=[5,10,50,28,7])[0]
        n_skills  = random.randint(2, 20)
        n_certs   = random.choices([0,1,2,3,4], weights=[40,30,15,10,5])[0]
        promoted  = random.random() < 0.45
        has_sum   = random.random() < 0.65
        has_li    = random.random() < 0.70
        has_gh    = random.random() < 0.50
        has_port  = random.random() < 0.30
        has_langs = random.random() < 0.55

        params = dict(
            yrs=yrs, n_jobs=n_jobs, edu_level=edu_level,
            n_skills=n_skills, n_certs=n_certs, promoted=promoted,
            has_summary=has_sum, has_linkedin=has_li,
            has_github=has_gh, has_portfolio=has_port, has_langs=has_langs,
        )
        records.append(build_resume(params))

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    print("Generating 3000 simulated resumes ...")
    df = simulate_dataset(3000)

    print(f"\nDataset shape   : {df.shape}")
    print(f"Score range     : {df['total_score'].min():.1f} – {df['total_score'].max():.1f}")
    print(f"Mean total score: {df['total_score'].mean():.1f}")
    print(f"\nSub-score means:")
    for col in ["experience_score","education_score","skills_score","structure_score"]:
        print(f"  {col:22s}: {df[col].mean():.2f} / 25")

    df.to_csv("data/resume_dataset.csv", index=False)
    print("\nSaved → data/resume_dataset.csv ✓")
    print("\nSample resume:\n" + "─"*60)
    print(df["resume_text"].iloc[0][:800])
