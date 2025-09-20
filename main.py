import os
import pandas as pd
from utils import extract_text
from hard_match import extract_skills_from_jd, hard_match_score
from semantic_match import semantic_score
from scoring import final_score, verdict

JD_FILE = "jd/sample_jd_1.pdf"
RESUMES_FOLDER = "resumes/"
OUTPUT_FILE = "outputs/resume_scores.csv"

jd_text = extract_text(JD_FILE)
jd_skills = extract_skills_from_jd(jd_text)

results = []

for resume_file in os.listdir(RESUMES_FOLDER):
    if not resume_file.lower().endswith(('.pdf', '.docx')):
        continue
    resume_path = os.path.join(RESUMES_FOLDER, resume_file)
    resume_text = extract_text(resume_path)
    
    hard_score, missing_skills = hard_match_score(resume_text, jd_skills)
    sem_score = semantic_score(resume_text, jd_text)
    score = final_score(hard_score, sem_score)
    fit = verdict(score)
    
    results.append({
        "Resume": resume_file,
        "Hard Score": round(hard_score,2),
        "Semantic Score": round(sem_score,2),
        "Final Score": round(score,2),
        "Fit Verdict": fit,
        "Missing Skills": ", ".join(missing_skills)
    })

df = pd.DataFrame(results)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Results saved to {OUTPUT_FILE}")