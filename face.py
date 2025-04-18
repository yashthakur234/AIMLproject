
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_groq import ChatGroq
# llm = ChatGroq(
#     temperature = 0,
#     groq_api_key = "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",
#     model_name = "llama-3.3-70b-versatile"
# )
# result = llm.invoke("Who is lord Ram?")
# print(result.content)




# # import gradio as gr
# # import os

# # # --- 1. Global initialization ---
# # llm = initialize_llm()
# # db_path = "/content/chroma_db"

# # if not os.path.exists(db_path):
# #     vector_db = create_vector_db()
# # else:
# #     embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# #     vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

# # qa_chain = setup_qa_chain(vector_db, llm)

# # # --- 2. Define chatbot logic function ---
# # def chatbot_fn(message, history):
# #     try:
# #         response = qa_chain.run(message)
# #         return response
# #     except Exception as e:
# #         return f"⚠️ Error: {str(e)}"

# # # --- 3. Custom CSS for background and style ---
# # custom_css = """
# # body {
# #     background: linear-gradient(135deg, #1a1a40, #0f2027);
# #     color: white;
# # }
# # .gradio-container {
# #     background-color: rgba(0, 0, 0, 0.6) !important;
# #     padding: 20px;
# #     border-radius: 16px;
# # }
# # h1, .title {
# #     color: #ffffff;
# #     text-align: center;
# #     font-weight: bold;
# #     margin-bottom: 20px;
# # }
# # .message.bot {
# #     background-color: rgba(255, 255, 255, 0.1) !important;
# #     color: white !important;
# # }
# # .message.user {
# #     background-color: rgba(255, 255, 255, 0.05) !important;
# #     color: white !important;
# # }
# # """

# # # --- 4. Launch with styled ChatInterface ---
# # chat_ui = gr.ChatInterface(
# #     fn=chatbot_fn,
# #     title="🧠 Mental Health Assistant",
# #     theme=gr.themes.Base(),  # You can also experiment with gr.themes.Soft()
# #     css=custom_css
# # )

# # chat_ui.launch()


# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# import gradio as gr
# def initialize_llm():
#   llm = ChatGroq(
#     temperature = 0,
#     groq_api_key = "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",
#     model_name = "llama-3.3-70b-versatile"
# )
#   return llm

# def create_vector_db():
#   loader = DirectoryLoader("/Users/yashpratapsingh/Desktop/upload/mental_health_Document.pdf", glob = '*.pdf', loader_cls = PyPDFLoader)
#   documents = loader.load()
#   text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
#   texts = text_splitter.split_documents(documents)
#   embeddings = HuggingFaceBgeEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
#   vector_db = Chroma.from_documents(texts, embeddings, persist_directory = './chroma_db')
#   vector_db.persist()

#   print("ChromaDB created and data saved")

#   return vector_db

# def setup_qa_chain(vector_db, llm):
#   retriever = vector_db.as_retriever()
#   prompt_templates = """ You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
#     {context}
#     User: {question}
#     Chatbot: """
#   PROMPT = PromptTemplate(template = prompt_templates, input_variables = ['context', 'question'])

#   qa_chain = RetrievalQA.from_chain_type(
#       llm = llm,
#       chain_type = "stuff",
#       retriever = retriever,
#       chain_type_kwargs = {"prompt": PROMPT}
#   )
#   return qa_chain


# print("Intializing Chatbot.........")
# llm = initialize_llm()

# db_path = "/content/chroma_db"

# if not os.path.exists(db_path):
#   vector_db  = create_vector_db()
# else:
#   embeddings = HuggingFaceBgeEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
#   vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
# qa_chain = setup_qa_chain(vector_db, llm)

# def chatbot_response(user_input, history = []):
#   if not user_input.strip():
#     return "Please provide a valid input", history
#   response = qa_chain.run(user_input)
#   history.append((user_input, response))
#   return "", history

# with gr.Blocks(theme = 'Respair/Shiki@1.2.1') as app:
#     gr.Markdown("# 🧠 Mental Health Chatbot 🤖")
#     gr.Markdown("A compassionate chatbot designed to assist with mental well-being. Please note: For serious concerns, contact a professional.")

#     chatbot = gr.ChatInterface(fn=chatbot_response, title="Mental Health Chatbot")

#     gr.Markdown("This chatbot provides general support. For urgent issues, seek help from licensed professionals.")

# app.launch()





from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os
import gradio as gr
import datetime
import pandas as pd
import numpy as np
from time import sleep
from pydub import AudioSegment
from pydub.playback import play

# --- Initialize Core Components ---
def initialize_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY", "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"),
        model_name="llama3-70b-8192"
    )

def create_vector_db():
    loader = DirectoryLoader("/Users/yashpratapsingh/Desktop/upload/", glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are a compassionate mental health assistant. Respond thoughtfully to:
    {context}
    Question: {question}
    Compassionate Response:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

# --- Mental Health Features ---
class MentalHealthFeatures:
    def __init__(self):
        self.mood_log = pd.DataFrame(columns=['timestamp', 'mood', 'intensity'])
        self.journal_entries = []
        
    def log_mood(self, mood, intensity):
        entry = {
            'timestamp': datetime.datetime.now(),
            'mood': mood,
            'intensity': intensity
        }
        self.mood_log = pd.concat([self.mood_log, pd.DataFrame([entry])], ignore_index=True)
        return "Mood logged successfully 🌱"
    
    def guided_meditation(self, duration):
        def update_progress(progress):
            return f"🕉️ Meditation Progress: {progress}% complete..."
        
        for i in range(1, 101):
            sleep(duration*0.6)
            yield update_progress(i)
        return "🧘♂️ Meditation session complete! Well done!"
    
    def breathing_exercise(self):
        stages = {
            "Breathe In (4s)": "🌬️💨",
            "Hold (4s)": "⏳",
            "Breathe Out (6s)": "😌"
        }
        for stage, emoji in stages.items():
            yield f"{emoji} {stage}"
            sleep(4 if "Out" not in stage else 6)
    
    def journal_entry(self, entry):
        self.journal_entries.append({
            'timestamp': datetime.datetime.now(),
            'entry': entry
        })
        return "📔 Journal entry saved successfully"
    
    def get_resources(self):
        return """🆘 Emergency Resources:
        - Suicide Hotline: 1-800-273-TALK (8255)
        - Crisis Text Line: Text 'HOME' to 741741
        - SAMHSA Helpline: 1-800-662-HELP (4357)"""

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), css="footer {visibility: hidden}") as app:
    mh_features = MentalHealthFeatures()
    llm = initialize_llm()
    
    # Initialize Vector DB
    if not os.path.exists("./chroma_db"):
        vector_db = create_vector_db()
    else:
        embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    qa_chain = setup_qa_chain(vector_db, llm)
    
    # UI Components
    gr.Markdown("# 🧠 Mental Health Companion 🤖")
    
    with gr.Tabs():
        with gr.TabItem("Chat"):
            chatbot = gr.ChatInterface(
                fn=lambda msg, _: qa_chain.run(msg),
                examples=["How to manage anxiety?", "What are coping strategies for stress?"],
                title="Mental Health Support"
            )
        
        with gr.TabItem("Wellness Tools"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🧘 Guided Meditation")
                    duration = gr.Slider(1, 30, value=5, label="Duration (minutes)")
                    meditation_output = gr.Textbox()
                    gr.Button("Start Meditation").click(
                        mh_features.guided_meditation,
                        inputs=[duration],
                        outputs=meditation_output
                    )
                    
                with gr.Column():
                    gr.Markdown("### 🌬️ Breathing Exercise")
                    breathing_output = gr.Textbox()
                    gr.Button("Start Breathing Exercise").click(
                        mh_features.breathing_exercise,
                        outputs=breathing_output
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 😌 Mood Tracking")
                    mood = gr.Dropdown(["Happy", "Sad", "Anxious", "Calm", "Angry"], label="Current Mood")
                    intensity = gr.Slider(1, 10, label="Intensity")
                    mood_output = gr.Textbox()
                    gr.Button("Log Mood").click(
                        mh_features.log_mood,
                        inputs=[mood, intensity],
                        outputs=mood_output
                    )
                
                with gr.Column():
                    gr.Markdown("### 📔 Journal")
                    journal_input = gr.Textbox(label="Today's Thoughts")
                    journal_output = gr.Textbox()
                    gr.Button("Save Entry").click(
                        mh_features.journal_entry,
                        inputs=[journal_input],
                        outputs=journal_output
                    )
        
        with gr.TabItem("Resources"):
            gr.Markdown(mh_features.get_resources())

# --- Deployment Instructions ---
"""
To deploy this chatbot:

1. Required Libraries (install via pip):
   - langchain
   - langchain-groq
   - gradio
   - chromadb
   - sentence-transformers
   - pydub
   - pandas
   - numpy

2. Deployment Options:
   A. Local Deployment:
      - Run: python app.py
      - Access via local URL in terminal

   B. Hugging Face Spaces:
      - Create new Space with Gradio SDK
      - Add secrets for GROQ_API_KEY
      - Upload code and requirements.txt

   C. Cloud Deployment (AWS/GCP):
      - Use Docker container with Python 3.9+
      - Environment variables:
        GROQ_API_KEY=your-api-key
      - Expose port 7860

3. Security Recommendations:
   - Use environment variables for API keys
   - Add rate limiting
   - Include disclaimer about professional help
"""
app.launch()