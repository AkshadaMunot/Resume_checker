import streamlit as st
import os
import tempfile
import pandas as pd
import sqlite3
from datetime import datetime

# Import your existing modules
from utils import extract_text, extract_location, extract_job_role
from hard_match import extract_skills_from_jd, hard_match_score
from scoring import final_score as compute_final_score, verdict as compute_verdict

# --- Set background image ---
def set_background_image(image_path):
    image_path = os.path.abspath(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("file://{image_path}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: rgba(255,255,255,0.25);  /* overlay transparency */
            z-index: 0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set your background image (replace 'background.jpg' with your file)
set_background_image("background.jpg")

# --- Embedding model loader (cached) ---
@st.cache_resource
def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

@st.cache_data
def semantic_score_with_model(resume_text, jd_text, model_name="all-MiniLM-L6-v2"):
    model = load_embedding_model(model_name)
    from sentence_transformers import util
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

# Sidebar: JD selection / upload
st.sidebar.header("Job Descriptions (JD)")
jd_choice = st.sidebar.radio("Load JD from:", ("Upload JD file(s)", "Use JD from jd/ folder"))

jd_texts = []  # list of tuples (filename, jd_text)

if jd_choice == "Upload JD file(s)":
    uploaded_jds = st.sidebar.file_uploader(
        "Upload JD(s) (PDF/DOCX)", type=["pdf","docx"], accept_multiple_files=True
    )
    if uploaded_jds:
        for jd_file in uploaded_jds:
            jd_path = save_uploaded_file(jd_file)
            text = extract_text(jd_path)
            jd_texts.append((jd_file.name, text))
else:
    jd_dir = "jd"
    os.makedirs(jd_dir, exist_ok=True)
    jd_files = [f for f in os.listdir(jd_dir) if f.lower().endswith((".pdf", ".docx"))]
    selected_files = st.sidebar.multiselect("Select JD file(s) from disk", jd_files)
    for f in selected_files:
        path = os.path.join(jd_dir, f)
        text = extract_text(path)
        jd_texts.append((f, text))

# Preview JDs
if jd_texts:
    st.subheader("Job Descriptions Preview")
    for jd_name, jd_text in jd_texts:
        with st.expander(f"JD: {jd_name}"):
            st.write(jd_text[:10000])
            jd_skills = extract_skills_from_jd(jd_text)
            job_role = extract_job_role(jd_text)
            st.markdown(f"**Job Role:** {job_role}")
            st.markdown("**Skills / Heuristics:**")
            st.write(jd_skills)
else:
    st.info("Upload or select at least one JD to proceed.")

# Sidebar: Scoring parameters
st.sidebar.header("Scoring parameters")
hard_weight = st.sidebar.slider("Hard-match weight", 0.0, 1.0, 0.6, step=0.05)
semantic_weight = st.sidebar.slider("Semantic-match weight", 0.0, 1.0, 0.4, step=0.05)
if abs(hard_weight + semantic_weight - 1.0) > 1e-6:
    st.sidebar.warning("Weights don't sum to 1 — they'll be normalized in computation.")

# Upload resumes
st.header("Upload Resumes")
uploaded_resumes = st.file_uploader(
    "Upload resumes (PDF / DOCX)", accept_multiple_files=True, type=["pdf","docx"]
)

run = st.button("Run evaluation")

# --- Evaluation ---
if run:
    if not jd_texts:
        st.error("No JD provided. Upload or select a JD first.")
    elif not uploaded_resumes:
        st.error("Upload at least one resume to evaluate.")
    else:
        all_results = []
        total = len(uploaded_resumes) * len(jd_texts)
        progress = st.progress(0)
        count = 0
        model_name = "all-MiniLM-L6-v2"

        for jd_name, jd_text in jd_texts:
            jd_skills = extract_skills_from_jd(jd_text)
            job_role = extract_job_role(jd_text)

            for up in uploaded_resumes:
                count += 1
                progress.progress(int(count / total * 100))

                resume_path = save_uploaded_file(up)
                resume_text = extract_text(resume_path)

                # Hard & semantic match
                hard_score, missing_skills = hard_match_score(resume_text, jd_skills)
                sem_score = semantic_score_with_model(resume_text, jd_text, model_name=model_name)

                # Normalize weights
                w_sum = hard_weight + semantic_weight
                hw = hard_weight / w_sum
                sw = semantic_weight / w_sum

                final = compute_final_score(hard_score, sem_score, hard_weight=hw, semantic_weight=sw)
                fit = compute_verdict(final)

                location = extract_location(resume_text)

                all_results.append({
                    "JD": jd_name,
                    "Resume": up.name,
                    "Job Role": job_role,
                    "Hard Score": round(hard_score,2),
                    "Semantic Score": round(sem_score,2),
                    "Final Score": round(final,2),
                    "Fit Verdict": fit,
                    "Location": location,
                    "Missing Skills": ", ".join(missing_skills)
                })

        st.success(f"Processed {len(uploaded_resumes)} resumes for {len(jd_texts)} JDs")

        # Dataframe
        df = pd.DataFrame(all_results).sort_values(["JD","Final Score"], ascending=[True,False])
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

        # Download results
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = f"outputs/resume_scores_{timestamp}.csv"
        df.to_csv(out_csv, index=False)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results (CSV)",
            data=csv_bytes,
            file_name=f"resume_scores_{timestamp}.csv",
            mime="text/csv"
        )
        st.markdown(f"Saved copy on server at: `{out_csv}`")

        # Charts: Top candidates per JD
        st.subheader("Top candidates by Final Score per JD")
        for jd_name in df["JD"].unique():
            jd_df = df[df["JD"] == jd_name].sort_values("Final Score", ascending=False)
            st.markdown(f"**JD:** {jd_name}")
            top_n = min(10, len(jd_df))
            st.bar_chart(jd_df.head(top_n).set_index("Resume")["Final Score"])

        # Filters
        st.subheader("Filter Candidates")
        selected_jds = st.multiselect("Filter by JD", options=df["JD"].unique(), default=df["JD"].unique())
        roles = df["Job Role"].unique().tolist()
        selected_roles = st.multiselect("Filter by Job Role", options=roles, default=roles)
        locations = df["Location"].unique().tolist()
        selected_locations = st.multiselect("Filter by Location", options=locations, default=locations)
        min_score = st.slider("Minimum Final Score", 0, 100, 50)

        filtered_df = df[
            (df["JD"].isin(selected_jds)) &
            (df["Job Role"].isin(selected_roles)) &
            (df["Location"].isin(selected_locations)) &
            (df["Final Score"] >= min_score)
        ]

        st.write(f"Showing {len(filtered_df)} candidates after filters")
        st.dataframe(filtered_df, use_container_width=True)

        # Detailed candidate view
        candidate = st.selectbox("Pick a candidate to inspect", options=df["Resume"].tolist())
        if candidate:
            row = df[df["Resume"] == candidate].iloc[0]
            st.markdown(f"**{candidate}** — Final Score: **{row['Final Score']}** — Verdict: **{row['Fit Verdict']}** — Location: **{row['Location']}**")
            st.markdown("**Missing skills**:")
            st.write(row["Missing Skills"] if row["Missing Skills"] else "None detected")

            for up in uploaded_resumes:
                if up.name == candidate:
                    tmp_path = save_uploaded_file(up)
                    text_preview = extract_text(tmp_path)[:5000]
                    with st.expander("Resume text preview (first 5k chars)"):
                        st.write(text_preview)
                    break

        # Save to SQLite
        if st.checkbox("Save results to local SQLite DB (outputs/resumes.db)"):
            conn = sqlite3.connect("outputs/resumes.db")
            df.to_sql("results", conn, if_exists="append", index=False)
            conn.close()
            st.success("Saved results to outputs/resumes.db (table: results)")

st.markdown("---")
st.caption("Notes: embeddings model is cached so the first run may be slow. Install `python-Levenshtein` to speed up fuzzy matching.")
