# model.py
import os
import PyPDF2
import pickle
import google.generativeai as genai

# Configure Gemini API with API key
GENMI_API_KEY = ""  # Replace with your Gemini API key
genai.configure(api_key=GENMI_API_KEY)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def load_cached_text(pdf_name):
    cache_path = os.path.join("temp_dir", f"{pdf_name}_extracted_text.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None


def save_extracted_text(pdf_name, extracted_text):
    cache_path = os.path.join("temp_dir", f"{pdf_name}_extracted_text.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(extracted_text, f)

# Function to query Gemini 1.5 Flash for question-answering
def query_gemini_api(question, context):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1600,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    # Start a chat session
    chat_session = model.start_chat(history=[])
    
    # Send message and get response
    full_context = context + "\n\n" + "Question: " + question
    response = chat_session.send_message(full_context)

    return response.text