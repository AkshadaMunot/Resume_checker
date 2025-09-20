from fuzzywuzzy import fuzz

def extract_skills_from_jd(jd_text):
    jd_text = jd_text.lower()
    skills = []

    # Simple heuristic: look for keywords "skills" or "requirements"
    if "skills" in jd_text:
        skills_section = jd_text.split("skills", 1)[1]
        skills = [s.strip() for s in skills_section.split(",") if len(s.strip()) > 1]

    # Fallback: define default list (modify as per your domain)
    if not skills:
        skills = ["python", "sql", "machine learning", "data analysis"]

    return skills

def hard_match_score(resume_text, jd_skills):
    if not jd_skills:  # Avoid division by zero
        return 0, []

    score = 0
    missing_skills = []
    for skill in jd_skills:
        ratio = fuzz.partial_ratio(skill.lower(), resume_text.lower())
        if ratio > 70:
            score += 1
        else:
            missing_skills.append(skill)
    total_score = (score / len(jd_skills)) * 100
    return total_score, missing_skills
