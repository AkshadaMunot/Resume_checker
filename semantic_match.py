from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_score(resume_text, jd_text):
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(resume_emb, jd_emb).item()
    return cosine_sim * 100  # scale 0-100