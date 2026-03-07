import streamlit as st
import spacy
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import pandas as pd
import altair as alt

# --- NLP Setup ---
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()
SKILLS_DB = [
    "Python", "Machine Learning", "NLP", "Flask", "SQL", "React", "Docker", 
    "AWS", "Java", "Tableau", "Git", "FastAPI", "C++", "Data Analysis", 
    "Deep Learning", "HTML", "CSS", "JavaScript", "Pandas", "NumPy", "Azure"
]

# --- Core Logic Functions ---
def clean_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def extract_skills(text):
    text_lower = text.lower()
    return [skill for skill in SKILLS_DB if skill.lower() in text_lower]

def get_text_from_pdf(file):
    file.seek(0)
    file_content = file.read()
    if not file_content: return ""
    doc = fitz.open(stream=file_content, filetype="pdf")
    text = "".join([page.get_text() for page in doc])
    return text

# --- Page Configuration ---
st.set_page_config(page_title="AI Talent Scout Pro", page_icon="🛡️", layout="wide")

# --- ULTRA NEON CUSTOM CSS ---
st.markdown("""
    <style>
    html { scroll-behavior: smooth; }
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
    
    * { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .logo, .hero-title { font-family: 'Orbitron', sans-serif; }
    
    /* Deep Black Background */
    .stApp { 
        background-color: #030303 !important; 
        color: #E2E8F0; 
    }

    /* Animated Neon Circles */
    .area {
        background: #030303;  
        width: 100%; height: 100vh;
        position: fixed; top: 0; left: 0;
        z-index: -1;
    }

    .circles li {
        position: absolute; display: block; list-style: none;
        width: 20px; height: 20px;
        background: rgba(0, 255, 153, 0.05); /* Neon Green Tint */
        border: 1px solid rgba(0, 255, 153, 0.2);
        animation: animate 25s linear infinite;
        bottom: -150px;
        box-shadow: 0 0 10px rgba(0, 255, 153, 0.2);
    }

    .circles li:nth-child(1){ left: 25%; width: 80px; height: 80px; animation-delay: 0s; }
    .circles li:nth-child(2){ left: 10%; width: 20px; height: 20px; animation-delay: 2s; animation-duration: 12s; }
    .circles li:nth-child(3){ left: 70%; width: 20px; height: 20px; animation-delay: 4s; }
    .circles li:nth-child(4){ left: 40%; width: 60px; height: 60px; animation-delay: 0s; animation-duration: 18s; }
    .circles li:nth-child(5){ left: 65%; width: 20px; height: 20px; animation-delay: 0s; }
    .circles li:nth-child(6){ left: 75%; width: 110px; height: 110px; animation-delay: 3s; }
    .circles li:nth-child(7){ left: 35%; width: 150px; height: 150px; animation-delay: 7s; }
    .circles li:nth-child(8){ left: 50%; width: 25px; height: 25px; animation-delay: 15s; animation-duration: 45s; }

    @keyframes animate {
        0%{ transform: translateY(0) rotate(0deg); opacity: 1; border-radius: 0; }
        100%{ transform: translateY(-1000px) rotate(720deg); opacity: 0; border-radius: 50%; }
    }

    /* Neon Navbar */
    .nav-header {
        display: flex; justify-content: space-between; align-items: center;
        padding: 1rem 4rem; background: rgba(0, 0, 0, 0.9);
        backdrop-filter: blur(15px); border-bottom: 2px solid #00FF99;
        box-shadow: 0 0 20px rgba(0, 255, 153, 0.3);
        position: fixed; top: 0; left: 0; right: 0; width: 100%; z-index: 999999;
    }
    .logo { font-weight: 900; font-size: 1.6rem; color: #00FF99; text-shadow: 0 0 10px #00FF99; text-decoration: none !important; }
    .nav-links a { text-decoration: none !important; color: #E2E8F0; font-weight: 600; transition: 0.3s; margin-left: 25px; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .nav-links a:hover { color: #00FF99; text-shadow: 0 0 8px #00FF99; }

    /* Neon Hero */
    .hero-container {
        text-align: center; padding: 180px 20px 100px 20px;
        background: radial-gradient(circle at center, rgba(0, 255, 153, 0.15) 0%, rgba(3, 3, 3, 0) 70%);
    }
    .hero-title { 
        font-size: 4.5rem; font-weight: 900; line-height: 1.1; margin-bottom: 20px;
        background: linear-gradient(90deg, #00FF99, #00D1FF, #BC00FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 15px rgba(0, 255, 153, 0.5));
    }

    .launch-btn-style {
        background: transparent; color: #00FF99; padding: 15px 50px; 
        border: 2px solid #00FF99; border-radius: 0px; display: inline-block; font-weight: bold; 
        font-family: 'Orbitron', sans-serif; text-transform: uppercase; letter-spacing: 2px;
        box-shadow: 0 0 15px rgba(0, 255, 153, 0.4);
        transition: 0.4s; cursor: pointer;
    }
    .launch-btn-style:hover { background: #00FF99; color: black; box-shadow: 0 0 30px #00FF99; transform: scale(1.05); }
    
    /* Neon Step Cards */
    .step-card {
        background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(0, 255, 153, 0.3);
        padding: 40px 25px; border-radius: 10px; text-align: center;
        transition: 0.4s; height: 100%; backdrop-filter: blur(10px);
    }
    .step-card:hover { 
        border-color: #00FF99; 
        box-shadow: 0 0 20px rgba(0, 255, 153, 0.3);
        background: rgba(0, 255, 153, 0.05);
    }
    .step-card h1 { filter: drop-shadow(0 0 10px currentColor); }
    .step-card h4 { color: #00D1FF; font-weight: 700; text-transform: uppercase; }

    /* Custom Input Styling */
    .stTextArea textarea { background: #000 !important; border: 1px solid #00FF99 !important; color: #00FF99 !important; box-shadow: inset 0 0 5px rgba(0, 255, 153, 0.2); }
    .stFileUploader { border: 1px dashed #00D1FF !important; background: #000 !important; }

    /* Expander Styling */
    .st-expander { background-color: rgba(255, 255, 255, 0.03) !important; border: 1px solid #333 !important; }

    header[data-testid="stHeader"] { background: transparent !important; }
    .section-anchor { padding-top: 130px; }

    /* Mobile Responsiveness Fix */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.2rem !important; }
        .nav-header { padding: 1rem !important; }
        .nav-links { display: none; } /* Hide nav links on mobile for space */
        .step-card { margin-bottom: 20px; }
        .stTable { overflow-x: auto; }
    }
            
            /* --- LABEL AND UPLOADER VISIBILITY FIX --- */
    
    /* 1. Fix for Black Labels (Job Description & Upload Resumes) */
    [data-testid="stWidgetLabel"] p, 
    [data-testid="stWidgetLabel"] div {
        color: #00FF99 !important; /* Neon Green text */
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* 2. File Uploader Container Fix */
    [data-testid="stFileUploader"] {
        background-color: #000 !important; /* Dark background */
        border: 1px dashed #00D1FF !important; /* Blue dashed border */
        padding: 15px;
        border-radius: 5px;
    }

    /* 3. Browse Files Button Visibility */
    [data-testid="stFileUploader"] button {
        background-color: #00FF99 !important; /* Bright Green */
        color: #000 !important; /* Black text for contrast */
        border: none !important;
        font-weight: bold !important;
        border-radius: 2px !important;
    }

    /* 4. Drag & Drop Text (Visible on live link) */
    [data-testid="stFileUploaderText"] div {
        color: #E2E8F0 !important;
    }
    
    /* Make Radar Chart responsive */
    .js-plotly-plot { width: 100% !important; }
            /* --- Force Dark Buttons for Live Link --- */
    .stButton>button {
        background-color: transparent !important;
        color: #00FF99 !important;
        border: 2px solid #00FF99 !important;
        border-radius: 0px !important;
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: 0.4s;
    }

    .stButton>button:hover {
        background-color: #00FF99 !important;
        color: #000 !important;
        box-shadow: 0 0 20px #00FF99 !important;
    }

    /* --- Mobile Responsiveness Enhancements --- */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.2rem !important; }
        .nav-header { padding: 1rem !important; }
        .nav-links { display: none; } 
        .step-card { margin-bottom: 20px; }
        
        /* Columns ko mobile par stack karne ke liye fix */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
        
        .hero-container { padding: 120px 10px 60px 10px !important; }
    }
    </style>
    
    <div class="area">
        <ul class="circles">
            <li></li><li></li><li></li><li></li><li></li>
            <li></li><li></li><li></li><li></li><li></li>
        </ul>
    </div>

    <div class="nav-header">
        <a href="#" class="logo">🛡️ RE-SCREEN AI</a>
        <div class="nav-links">
            <a href="#features">Features</a>
            <a href="#how-it-works">Protocol</a>
            <a href="#analysis-hub">Terminal</a>
            <a href="https://github.com" target="_blank" style="border: 1px solid #00FF99; color: #00FF99; padding: 6px 15px; border-radius: 4px;">GitHub</a>
        </div>
    </div>
    
    <div class="hero-container">
        <span class="hero-title">NEURAL RESUME<br>SCREENING</span>
        <p style="color: #00D1FF; font-size: 1.1rem; max-width: 650px; margin: 0 auto; letter-spacing: 2px; text-transform: uppercase; font-weight: bold;">
            [ SYSTEM STATUS: READY ] <br> Advanced NLP Extraction • Vector Matrix Mapping
        </p>
        <br><br>
        <a href="#analysis-hub" style="text-decoration: none;">
            <div class="launch-btn-style">Initialize Scan</div>
        </a>
    </div>
    """, unsafe_allow_html=True)

# --- Features & Workflow ---
st.markdown('<div id="features" class="section-anchor"></div>', unsafe_allow_html=True) 
st.markdown("<h2 style='text-align: center; color: #00FF99; text-shadow: 0 0 10px rgba(0,255,153,0.5);'>CORE SYSTEM MODULES</h2>", unsafe_allow_html=True)
f1, f2, f3 = st.columns(3)
with f1: st.markdown('<div class="step-card"><h4>🔍 NLP Analysis</h4><p>Context-aware entity recognition via Spacy Neural Models.</p></div>', unsafe_allow_html=True)
with f2: st.markdown('<div class="step-card"><h4>📊 Vector Engine</h4><p>High-precision TF-IDF Cosine Similarity Matrix.</p></div>', unsafe_allow_html=True)
with f3: st.markdown('<div class="step-card"><h4>🎯 Gap Protocol</h4><p>Instant delta detection between JD and Candidate skills.</p></div>', unsafe_allow_html=True)

st.markdown('<div id="how-it-works" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #00D1FF; text-shadow: 0 0 10px rgba(0,209,255,0.5);'>OPERATIONAL WORKFLOW</h2>", unsafe_allow_html=True)
h1, h2, h3, h4 = st.columns(4)
steps = [("💾", "Data Input", "Inject multiple PDF profiles"), ("🧹", "Refining", "AI-based Noise Cancellation"), ("🧬", "Synthesis", "Vector Logic Processing"), ("💹", "Reporting", "Real-time Meta Ranking")]
for i, (icon, title, desc) in enumerate(steps):
    with [h1, h2, h3, h4][i]: st.markdown(f'<div class="step-card"><h1>{icon}</h1><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

# --- PROCESSING HUB ---
st.markdown('<div id="analysis-hub" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown("<br><hr style='border:0.1px solid #00FF99; opacity:0.3;'><br>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#00FF99; font-family:Orbitron;'>// AI PROCESSING TERMINAL</h3>", unsafe_allow_html=True)

col_jd, col_file = st.columns(2, gap="large")
with col_jd:
    jd_text = st.text_area("📝 Job Description (Requirement)", height=230, placeholder="Input system requirements...")
with col_file:
    uploaded_files = st.file_uploader("📄 Upload Resumes (PDF Source)", type=["pdf"], accept_multiple_files=True)

if st.button("EXECUTE ANALYSIS"):
    if jd_text and uploaded_files:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        c_jd = clean_text(jd_text)
        jd_s = extract_skills(jd_text)

        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Scanning Node: {file.name}...")
            resume_raw = get_text_from_pdf(file)
            c_res = clean_text(resume_raw)
            res_s = extract_skills(resume_raw)
            
            cv = TfidfVectorizer()
            matrix = cv.fit_transform([c_res, c_jd])
            base_score = cosine_similarity(matrix)[0][1] * 100
            
            matched = list(set(jd_s) & set(res_s))
            missing = list(set(jd_s) - set(res_s))
            skill_score = (len(matched) / len(jd_s) * 100) if jd_s else 0
            
            final_score = round(min((base_score * 0.5) + (skill_score * 0.5), 100.0), 2)
            results.append({"Candidate": file.name, "Score": final_score, "Matched": matched, "Missing": missing})
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
            time.sleep(0.05)
        
        status_text.markdown("<span style='color:#00FF99;'>SYSTEM STATUS: ANALYSIS COMPLETE</span>", unsafe_allow_html=True)
        ranked_df = pd.DataFrame(results).sort_values(by="Score", ascending=True)

        st.markdown("<br><h3 style='color:#00D1FF; font-family:Orbitron;'>// ANALYTICAL METRICS</h3>", unsafe_allow_html=True)
        
        # --- NEON TRADING CHART ---
        base = alt.Chart(ranked_df).encode(
            x=alt.X('Candidate:N', sort='-y', title=None, axis=alt.Axis(labelAngle=-45, labelColor='#00D1FF', tickColor='#00FF99')),
            tooltip=['Candidate', 'Score']
        )

        bars = base.mark_bar(size=40, cornerRadiusTopLeft=0, cornerRadiusTopRight=0, color='#00FF99', opacity=0.3, stroke='#00FF99', strokeWidth=1).encode(
            y=alt.Y('Score:Q', title='Match %', scale=alt.Scale(domain=[0, 110]))
        )

        line = base.mark_line(color='#BC00FF', strokeWidth=4, interpolate='monotone').encode(y='Score:Q')
        points = base.mark_point(color='#BC00FF', filled=True, size=200, stroke='white', strokeWidth=2).encode(y='Score:Q')

        trading_chart = alt.layer(bars, line, points).properties(height=450, background='transparent').configure_view(strokeOpacity=0).configure_axis(gridColor='#222', gridOpacity=0.5, domain=False)

        st.altair_chart(trading_chart, use_container_width=True)

        st.markdown("<h2 style='color:#00FF99; font-family:Orbitron;'>// RANKED LEADERBOARD</h2>", unsafe_allow_html=True)
        for rank, res in enumerate(ranked_df.sort_values(by="Score", ascending=False).to_dict('records')):
            medal = "⚡" if rank == 0 else "🔹"
            with st.expander(f"{medal} CANDIDATE_RANK_{rank+1}: {res['Candidate']} — MATCH_{res['Score']}%"):
                c1, c2 = st.columns(2)
                with c1: st.success(f"**SKILLS_DETECTED:** {', '.join(res['Matched']) if res['Matched'] else 'None'}")
                with c2: st.error(f"**DELTAS_MISSING:** {', '.join(res['Missing']) if res['Missing'] else 'None'}")

        st.download_button("📥 DOWNLOAD ENCRYPTED REPORT", ranked_df.to_csv(index=False), "system_report.csv")
    else:
        st.warning("SYSTEM ERROR: Please provide Data Input (JD & Resumes).")

# --- CYBER FOOTER ---
st.markdown("""
    <div style="background: rgba(0, 255, 153, 0.05); border-top: 2px solid #00FF99; padding: 60px 40px; margin-top: 100px;">
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 300px;">
                <h4 style="color:#00FF99; font-family:Orbitron;">🛡️ RE-SCREEN AI</h4>
                <p style="color:#888;">Protocol version: 3.0.4.5<br>AI-Driven Meta Screening for high-tier recruitment.</p>
            </div>
            <div style="flex: 1; text-align: right;">
                <p style="color:#00D1FF; font-family:Orbitron;">SYSTEM DEVELOPER: VANSHIKA GUPTA</p>
                <p style="color:#444;">© 2026 ALL RIGHTS RESERVED | ENCRYPTED CONNECTION</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
