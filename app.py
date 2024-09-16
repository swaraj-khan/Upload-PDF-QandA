import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
import google.generativeai as genai
import re
import datetime

# Configure the Generative AI model with an API key
genai.configure(api_key="AIzaSyCj1kqcEZjM51RHbIASsM7GEvh889CDnb4")
model = genai.GenerativeModel('gemini-1.5-flash')

def sanitize_input(input_text):
    return re.sub(r'[^\w\s,.;!?-]', '', input_text)

def extract_questions_answers(files):
    qa_dict = {}
    for file in files:
        reader = PyPDF2.PdfReader(file)
        current_question = None
        collecting_answer = False
        answer_lines = []
        for page in reader.pages:
            text = page.extract_text() or ''
            lines = text.split('\n')
            for line in lines:
                if line.startswith("Q"):
                    if current_question and answer_lines:
                        qa_dict[current_question] = " ".join(answer_lines)
                    current_question = line.split(":")[1].strip()
                    collecting_answer = True
                    answer_lines = []
                elif collecting_answer:
                    if line.strip():
                        answer_lines.append(line.strip())
        if current_question and answer_lines:
            qa_dict[current_question] = " ".join(answer_lines)
    return qa_dict

def encode_questions(questions):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return {question: model.encode([question])[0] for question in questions}

def find_answer(question, question_embeddings, qa_dict):
    question_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode([question])[0]
    best_question = None
    max_cosine = -1
    for q, embedding in question_embeddings.items():
        cosine = (question_embedding @ embedding) / (np.linalg.norm(question_embedding) * np.linalg.norm(embedding))
        if cosine > max_cosine:
            max_cosine = cosine
            best_question = q

    if max_cosine >= 0.7:
        return qa_dict[best_question]
    else:
        return "No question & answer found. Try again with a new question."

def save_data(data, filename):
    with open(filename, "wb") as file:
        pickle.dump(data, file)

def load_data(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

def log_interaction(question, answer):
    with open("interaction_log.txt", "a") as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{timestamp} - Question: {question} | Answer: {answer}\n")

# Set up the Streamlit page
st.set_page_config(page_title="Q&A Chatbot", page_icon=":books:", layout="wide")
st.title("🤖 Q&A Chatbot")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# File uploader to accept a single PDF file
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if pdf_file:
    with st.spinner('Processing the PDF...'):
        qa_dict = extract_questions_answers([pdf_file])  # Accepting a single PDF file from the user
        question_embeddings = encode_questions(qa_dict.keys())
        save_data({'qa_dict': qa_dict, 'embeddings': question_embeddings}, "qa_embeddings.pkl")
    st.success('PDF processed successfully!')

if "history" not in st.session_state:
    st.session_state.history = []

def display_chat():
    if st.session_state.history:
        for qa in st.session_state.history:
            question_markdown = f"**Q:** {qa['question']}"
            answer_markdown = f"**A:**\n```python\n{qa['answer']}\n```"
            st.markdown(question_markdown, unsafe_allow_html=True)
            st.markdown(answer_markdown, unsafe_allow_html=True)
            st.markdown('<hr>', unsafe_allow_html=True)
    question = st.text_input("Ask a question:", key="new_question")
    if question and (not st.session_state.get('last_question') or st.session_state.last_question != question):
        sanitized_question = sanitize_input(question)
        data = load_data("qa_embeddings.pkl")
        answer = find_answer(sanitized_question, data['embeddings'], data['qa_dict'])
        log_interaction(sanitized_question, answer)
        st.session_state.history.append({"question": sanitized_question, "answer": answer})
        st.session_state.last_question = question
        st.rerun()

display_chat()
