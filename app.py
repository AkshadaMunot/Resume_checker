# app.py
import streamlit as st
import os
import tempfile
import pandas as pd
import sqlite3
from datetime import datetime

# Import your existing modules
from utils import extract_text
from hard_match import extract_skills_from_jd, hard_match_score
from scoring import final_score as compute_final_score, verdict as compute_verdict

# --- Embedding model loader (cached so it loads only once per session) ---
@st.cache_resource
def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

@st.cache_data
def semantic_score_with_model(resume_text, jd_text, model_name="all-MiniLM-L6-v2"):
    model = load_embedding_model(model_name)
    from sentence_transformers import util
    # encode (convert_to_tensor=True uses pytorch tensors)
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(resume_emb, jd_emb).item()
    return float(cosine_sim * 100)  # scale 0-100

# --- Helper: write uploaded file to a temp file and return path ---
def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name

# --- App UI ---
st.set_page_config(page_title="Resume Relevance — Dashboard", layout="wide")
st.title("Automated Resume Relevance Check — Dashboard")

# Sidebar: JD selection / upload and parameters
st.sidebar.header("Job Description (JD)")
jd_choice = st.sidebar.radio("Load JD from:", ("Upload JD file", "Use JD from jd/ folder"))

jd_text = ""
selected_jd_path = None

if jd_choice == "Upload JD file":
    uploaded_jd = st.sidebar.file_uploader("Upload JD (PDF/DOCX)", type=["pdf", "docx"])
    if uploaded_jd is not None:
        selected_jd_path = save_uploaded_file(uploaded_jd)
        jd_text = extract_text(selected_jd_path)
else:
    # list files in jd/ folder
    jd_dir = "jd"
    os.makedirs(jd_dir, exist_ok=True)
    jd_files = [f for f in os.listdir(jd_dir) if f.lower().endswith((".pdf", ".docx"))]
    selected = st.sidebar.selectbox("Select a JD file from disk", ["-- choose --"] + jd_files)
    if selected != "-- choose --":
        selected_jd_path = os.path.join(jd_dir, selected)
        jd_text = extract_text(selected_jd_path)

# Show JD preview and extracted skills
if jd_text:
    st.subheader("Job Description preview")
    with st.expander("Show JD text"):
        st.write(jd_text[:10000])  # first chunk
    jd_skills = extract_skills_from_jd(jd_text)
    st.markdown("**Extracted Skills / Heuristics:**")
    st.write(jd_skills)
else:
    st.info("Upload or select a job description (JD) to proceed.")

# Sidebar: Scoring parameters
st.sidebar.header("Scoring parameters")
hard_weight = st.sidebar.slider("Hard-match weight", 0.0, 1.0, 0.6, step=0.05)
semantic_weight = st.sidebar.slider("Semantic-match weight", 0.0, 1.0, 0.4, step=0.05)
# Ensure weights sum to 1 (optional)
if abs(hard_weight + semantic_weight - 1.0) > 1e-6:
    st.sidebar.warning("Weights don't sum to 1 — they'll be normalized in computation.")

# Upload resumes
st.header("Upload Resumes")
uploaded_resumes = st.file_uploader("Upload resumes (PDF / DOCX) — you can select multiple", accept_multiple_files=True, type=["pdf","docx"])

run = st.button("Run evaluation")  # triggers processing

if run:
    if not jd_text:
        st.error("No JD provided. Upload or select a JD first.")
    elif not uploaded_resumes:
        st.error("Upload at least one resume to evaluate.")
    else:
        # Prepare output list
        results = []
        progress = st.progress(0)
        total = len(uploaded_resumes)
        model_name = "all-MiniLM-L6-v2"

        for i, up in enumerate(uploaded_resumes, start=1):
            progress.progress(int((i-1)/total*100))
            # save resume to temp file
            resume_path = save_uploaded_file(up)
            resume_text = extract_text(resume_path)

            # Hard match
            hard_score, missing_skills = hard_match_score(resume_text, jd_skills)

            # Semantic match (cached)
            sem_score = semantic_score_with_model(resume_text, jd_text, model_name=model_name)

            # Normalize weights if not summing to 1
            w_sum = hard_weight + semantic_weight
            hw = hard_weight / w_sum
            sw = semantic_weight / w_sum

            final = compute_final_score(hard_score, sem_score, hard_weight=hw, semantic_weight=sw)
            fit = compute_verdict(final)

            results.append({
                "Resume": up.name,
                "Hard Score": round(hard_score,2),
                "Semantic Score": round(sem_score,2),
                "Final Score": round(final,2),
                "Fit Verdict": fit,
                "Missing Skills": ", ".join(missing_skills)
            })

        progress.progress(100)
        st.success(f"Processed {total} resumes")

        # Dataframe
        df = pd.DataFrame(results).sort_values("Final Score", ascending=False)
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

        # Download results & save to outputs
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = f"outputs/resume_scores_{timestamp}.csv"
        df.to_csv(out_csv, index=False)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results (CSV)", data=csv_bytes, file_name=f"resume_scores_{timestamp}.csv", mime="text/csv")
        st.markdown(f"Saved copy on server at: `{out_csv}`")

        # Simple bar chart of top 10
        top_n = min(10, len(df))
        st.subheader(f"Top {top_n} candidates by Final Score")
        st.bar_chart(df.head(top_n).set_index("Resume")["Final Score"])

        # Filtering and details
        st.subheader("Filter & Inspect")
        min_score = st.slider("Minimum final score to show", 0.0, 100.0, 50.0)
        verdict_filter = st.multiselect("Verdict filter", options=list(df["Fit Verdict"].unique()), default=list(df["Fit Verdict"].unique()))
        filtered = df[(df["Final Score"] >= min_score) & (df["Fit Verdict"].isin(verdict_filter))]
        st.write(f"Showing {len(filtered)} candidates after filter")
        st.dataframe(filtered, use_container_width=True)

        # Detailed candidate view
        candidate = st.selectbox("Pick a candidate to inspect", options=df["Resume"].tolist())
        if candidate:
            row = df[df["Resume"] == candidate].iloc[0]
            st.markdown(f"**{candidate}** — Final Score: **{row['Final Score']}** — Verdict: **{row['Fit Verdict']}**")
            st.markdown("**Missing skills**:")
            st.write(row["Missing Skills"] if row["Missing Skills"] else "None detected")

            # Show resume text preview (read temp file saved earlier)
            # find matching uploaded file
            for up in uploaded_resumes:
                if up.name == candidate:
                    tmp_path = save_uploaded_file(up)  # create temp copy to read
                    text_preview = extract_text(tmp_path)[:5000]
                    with st.expander("Resume text preview (first 5k chars)"):
                        st.write(text_preview)
                    break

        # Optional: Save to sqlite (toggle)
        if st.checkbox("Save results to local SQLite DB (outputs/resumes.db)"):
            conn = sqlite3.connect("outputs/resumes.db")
            df.to_sql("results", conn, if_exists="append", index=False)
            conn.close()
            st.success("Saved results to outputs/resumes.db (table: results)")

st.markdown("---")
st.caption("Notes: embeddings model is cached so the first run may be slow. Make sure `python-Levenshtein` is installed to speed up fuzzy matching.")
