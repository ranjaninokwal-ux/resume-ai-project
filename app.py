
# ===== IMPORTS =====
import PyPDF2, pandas as pd, gradio as gr, matplotlib.pyplot as plt, requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ===== DATABASE (SIMULATED) =====
users = {"admin":"1234"}
user_history = {}
current_user = {"name": None}

# ===== SIGNUP =====
def signup(username, password):
    if username in users:
        return "User already exists ❌"
    users[username] = password
    user_history[username] = []
    return "Signup successful ✅"

# ===== LOGIN =====
def login(username, password):
    if username in users and users[username] == password:
        current_user["name"] = username
        if username not in user_history:
            user_history[username] = []
        return f"Welcome {username} 🎉"
    return "Invalid login ❌"

# ===== REAL JOB API =====
def fetch_jobs():
    try:
        url = "https://remotive.com/api/remote-jobs"
        data = requests.get(url).json()
        jobs = [(j["title"], j["category"]) for j in data["jobs"][:20]]
        return jobs
    except:
        return [("Data Scientist","AI"),("Web Developer","IT")]

# ===== ANALYSIS =====
def analyze(file):
    if current_user["name"] is None:
        return "Login first", None, "", ""

    reader = PyPDF2.PdfReader(file)
    text = ""
    for p in reader.pages:
        text += p.extract_text()
    text = text.lower()

    jobs = fetch_jobs()
    job_texts = [j[0] + " " + j[1] for j in jobs]

    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(job_texts + [text])
    sim = cosine_similarity(tfidf[-1], tfidf[:-1])[0]

    results = sorted([(jobs[i][0], round(sim[i]*100,2)) for i in range(len(jobs))],
                     key=lambda x: x[1], reverse=True)[:5]

    user_history[current_user["name"]].append(results)

    # SCORE
    keywords = ["python","project","internship","skill"]
    score = int((sum([1 for k in keywords if k in text])/len(keywords))*100)

    # CHART
    names = [j for j,_ in results]
    scores = [s for _,s in results]
    plt.figure()
    plt.barh(names, scores)
    plt.title("Top Matches")

    # SUGGESTIONS
    sug = []
    if "project" not in text: sug.append("Add projects")
    if "internship" not in text: sug.append("Add internship")

    return f"{score}%", plt, "\n".join([f"{j}-{s}%" for j,s in results]), "\n".join(sug)

# ===== CHATBOT =====
def chatbot(msg):
    msg = msg.lower()
    if "resume" in msg:
        return "Add skills, projects, achievements."
    elif "job" in msg:
        return "Focus on job-specific skills."
    return "Ask about resume or career."

# ===== PDF REPORT =====
def generate_pdf(text):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = [
        Paragraph("Resume Analysis Report", styles["Title"]),
        Paragraph(text, styles["Normal"])
    ]
    doc.build(content)
    return "report.pdf"

# ===== SHOW HISTORY =====
def show_history():
    return str(user_history)

# ===== UI =====
with gr.Blocks(theme=gr.themes.Glass()) as demo:

    gr.Markdown("# 🚀 AI Resume Analyzer PRO MAX")

    # SIGNUP
    gr.Markdown("## 🆕 New User Signup")
    su_user = gr.Textbox(label="New Username")
    su_pass = gr.Textbox(label="New Password", type="password")
    su_btn = gr.Button("Sign Up")
    su_out = gr.Textbox()

    su_btn.click(signup, [su_user, su_pass], su_out)

    # LOGIN
    gr.Markdown("## 🔐 Login")
    li_user = gr.Textbox(label="Username")
    li_pass = gr.Textbox(label="Password", type="password")
    li_btn = gr.Button("Login")
    li_out = gr.Textbox()

    li_btn.click(login, [li_user, li_pass], li_out)

    # ANALYSIS
    gr.Markdown("## 📄 Upload Resume")
    file = gr.File()
    btn = gr.Button("Analyze")

    score = gr.Textbox(label="Resume Score")
    chart = gr.Plot()
    jobs_out = gr.Textbox(label="Top Job Matches")
    suggestions = gr.Textbox(label="Suggestions")

    btn.click(analyze, file, [score, chart, jobs_out, suggestions])

    # PDF
    pdf_btn = gr.Button("Download Report")
    pdf_file = gr.File()
    pdf_btn.click(generate_pdf, jobs_out, pdf_file)

    # CHATBOT
    gr.Markdown("## 🤖 Chatbot")
    msg = gr.Textbox(label="Ask something")
    reply = gr.Textbox()
    gr.Button("Send").click(chatbot, msg, reply)

    # HISTORY
    gr.Markdown("## 📊 User History")
    history = gr.Textbox()
    gr.Button("Show History").click(show_history, None, history)

demo.launch(server_name="0.0.0.0", server_port=10000)
