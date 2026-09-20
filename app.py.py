import json
import re
import pandas as pd
import PyPDF2
import streamlit as st
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Interview Prep & Mock Coach",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Extracts raw text stream from an uploaded PDF resume."""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + " "
        return re.sub(r"\s+", " ", extracted_text).strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""


def get_gemini_client(api_key: str):
    """Initializes and returns the Google Gemini Client."""
    return genai.Client(api_key=api_key)


def generate_interview_questions(client, role: str, experience: str, domain: str, resume_text: str, num_questions: int = 3):
    """Generates structured interview questions using Gemini AI."""
    prompt = f"""
    You are an expert technical interviewer hiring for the position of '{role}'.
    Candidate Experience Level: {experience}
    Domain / Tech Stack: {domain}
    Candidate Resume Summary: {resume_text if resume_text else 'Not Provided'}

    Generate exactly {num_questions} relevant interview questions.
    Return ONLY a JSON array of strings containing the questions.
    Example output format:
    ["Question 1 text here...", "Question 2 text here...", "Question 3 text here..."]
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.7,
        ),
    )
    return json.loads(response.text)


def evaluate_candidate_answer(client, question: str, candidate_answer: str, role: str):
    """Evaluates candidate response across technical correctness, clarity, and feedback."""
    prompt = f"""
    You are an AI Interview Evaluator assessing an answer for a '{role}' position.

    Interview Question: "{question}"
    Candidate Answer: "{candidate_answer}"

    Evaluate the answer and return ONLY a JSON object with the following fields:
    - "score": An integer score from 1 to 10.
    - "strengths": A short string detailing what the candidate answered well.
    - "improvements": A short string detailing missing key technical terms, gaps, or areas for improvement.
    - "model_answer": A clear, professional sample response demonstrating how to answer effectively using industry standards (or STAR method if behavioral).

    Example JSON output format:
    {{
        "score": 8,
        "strengths": "Good explanation of core syntax and clear logic.",
        "improvements": "Missed mentioning edge cases and memory efficiency.",
        "model_answer": "An ideal answer should highlight..."
    }}
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
        ),
    )
    return json.loads(response.text)


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_q_index" not in st.session_state:
    st.session_state.current_q_index = 0
if "evaluation_history" not in st.session_state:
    st.session_state.evaluation_history = []
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

# -----------------------------------------------------------------------------
# APPLICATION HEADER & SIDEBAR CONFIGURATION
# -----------------------------------------------------------------------------
st.title("🤖 AI Interview Prep & Mock Coach")
st.markdown("Dynamic question generation, automated answer scoring, and real-time feedback driven by Generative AI.")
st.divider()

# Sidebar: API Key Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    gemini_api_key = st.text_input("Enter Gemini API Key:", type="password")
    st.info("Get a free API key from [Google AI Studio](https://aistudio.google.com/).")
    
    st.divider()
    if st.button("🔄 Reset Interview Session"):
        st.session_state.questions = []
        st.session_state.current_q_index = 0
        st.session_state.evaluation_history = []
        st.session_state.interview_started = False
        st.rerun()

# -----------------------------------------------------------------------------
# STEP 1: INTERVIEW SETUP & CONTEXT INGESTION
# -----------------------------------------------------------------------------
if not st.session_state.interview_started:
    st.subheader("1. Setup Your Mock Interview")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        target_role = st.text_input("Target Job Role:", placeholder="e.g., Python Developer, Data Analyst, HR Manager")
        experience_level = st.selectbox("Experience Level:", ["Entry Level / Fresher", "Mid Level (2-4 Years)", "Senior Level (5+ Years)"])
        tech_domain = st.text_input("Key Skills / Domain Focus:", placeholder="e.g., Django, SQL, Machine Learning, Communication")
        num_q = st.slider("Number of Questions:", min_value=1, max_value=5, value=3)

    with col2:
        uploaded_resume = st.file_uploader("Upload Resume (PDF - Optional):", type=["pdf"])
        resume_text = ""
        if uploaded_resume is not None:
            resume_text = extract_text_from_pdf(uploaded_resume)
            st.success("Resume parsed successfully!")
            with st.expander("Preview Extracted Resume Text"):
                st.write(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)

    if st.button("🚀 Start Mock Interview", type="primary"):
        if not gemini_api_key.strip():
            st.error("Please provide a valid Gemini API Key in the sidebar.")
        elif not target_role.strip():
            st.error("Please enter a Target Job Role.")
        else:
            with st.spinner("Analyzing profile & generating customized interview questions..."):
                try:
                    client = get_gemini_client(gemini_api_key)
                    questions = generate_interview_questions(
                        client, target_role, experience_level, tech_domain, resume_text, num_q
                    )
                    st.session_state.questions = questions
                    st.session_state.interview_started = True
                    st.session_state.current_q_index = 0
                    st.session_state.evaluation_history = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate questions: {e}")

# -----------------------------------------------------------------------------
# STEP 2: ACTIVE MOCK INTERVIEW SESSION
# -----------------------------------------------------------------------------
else:
    total_q = len(st.session_state.questions)
    curr_idx = st.session_state.current_q_index

    if curr_idx < total_q:
        st.subheader(f"Question {curr_idx + 1} of {total_q}")
        current_question = st.session_state.questions[curr_idx]
        
        st.info(f"**Interviewer:** {current_question}")

        candidate_response = st.text_area(
            "Type your response below:",
            height=200,
            placeholder="Structure your answer clearly. Explain key terms, steps, or use the STAR method for behavioral questions...",
        )

        col_btn1, col_btn2 = st.columns([1, 4])
        
        with col_btn1:
            submit_answer = st.button("Submit Answer 📤", type="primary")

        if submit_answer:
            if not candidate_response.strip():
                st.warning("Please enter a response before submitting.")
            else:
                with st.spinner("AI Evaluator is analyzing your response..."):
                    try:
                        client = get_gemini_client(gemini_api_key)
                        evaluation = evaluate_candidate_answer(
                            client, current_question, candidate_response, "Candidate"
                        )
                        
                        # Save evaluation to history
                        st.session_state.evaluation_history.append({
                            "question": current_question,
                            "user_answer": candidate_response,
                            "score": evaluation.get("score", 0),
                            "strengths": evaluation.get("strengths", "N/A"),
                            "improvements": evaluation.get("improvements", "N/A"),
                            "model_answer": evaluation.get("model_answer", "N/A")
                        })
                        
                        st.session_state.current_q_index += 1
                        st.rerun()
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")

    # -------------------------------------------------------------------------
    # STEP 3: FINAL SCORE & PERFORMANCE REPORT
    # -------------------------------------------------------------------------
    else:
        st.balloons()
        st.success("🎉 Mock Interview Completed!")
        st.subheader("📊 Performance Summary & Feedback Report")

        history = st.session_state.evaluation_history
        df_history = pd.DataFrame(history)

        avg_score = round(df_history["score"].mean(), 2) if not df_history.empty else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Questions Attempted", len(history))
        m2.metric("Average Score", f"{avg_score} / 10")
        m3.metric("Performance Status", "High Readiness" if avg_score >= 7 else ("Moderate" if avg_score >= 5 else "Needs Practice"))

        st.divider()

        # Detailed Q&A Breakdown
        for i, item in enumerate(history):
            with st.expander(f"Question {i+1}: {item['question']} — Score: {item['score']}/10"):
                st.markdown(f"**Your Answer:**\n{item['user_answer']}")
                st.markdown(f"**✅ Strengths:** {item['strengths']}")
                st.markdown(f"**⚠️ Areas for Improvement:** {item['improvements']}")
                st.markdown(f"**💡 Model Answer:**\n{item['model_answer']}")

        st.divider()
        st.dataframe(df_history[["question", "score"]], use_container_width=True)
