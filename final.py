import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Set up API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
groq_client = Groq()  # Correctly initialize Groq client

# Initialize memory
memory = ConversationBufferMemory()

# Streamlit app title and description
st.title("Conversational AI Chatbot")
st.subheader("Chat with an AI model of your choice")

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Custom CSS for chat bubbles
st.markdown(
    """
    <style>
    .user-bubble {
        background-color: #007bff;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .bot-bubble {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 80%;
        color: white;
        float: left;
        clear: both;
    }
    .clearfix {
        overflow: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the conversation history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.markdown(f'<div class="clearfix"><div class="user-bubble">User: {msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="clearfix"><div class="bot-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)

# CSS to keep the input field always visible
st.markdown(
    """
    <style>
    .fixed-bottom-input {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0e1117;
        padding: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Placeholder for the input field with the fixed class
input_container = st.empty()

def submit():
    # Define max_tokens and temperature
    max_tokens = 4000
    temperature = 0.02

    # Append user input to session state history
    st.session_state.history.append({"role": "user", "content": st.session_state.user_input})

    # Retrieve the previous conversation history
    previous_conversation = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.history])
    
    # Construct the input prompt
    input_prompt = f"{previous_conversation}\nYou: {st.session_state.user_input}"
    
    # Initialize the appropriate model
    model_choice = st.session_state.get('model_choice', 'GPT-4')
    
    if model_choice == "GPT-4":
        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": input_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stop=None
        )
        generated_text = completion.choices[0].message.content

    elif model_choice == "Groq LLama-3.1":
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": input_prompt}],
            model="llama-3.1-8b-instant"
        )
        generated_text = response.choices[0].message.content

    elif model_choice == "GPT-4o":
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": input_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stop=None
        )
        generated_text = completion.choices[0].message.content

    # Append bot response to session state history
    st.session_state.history.append({"role": "bot", "content": generated_text})

    # Clear the input field
    st.session_state.user_input = ""

# Render the input field in the fixed position
with input_container.container():
    # Input text box
    st.text_input("Type your message here and press Enter:", key="user_input", placeholder="Type your message here...")

    # Display model selector and submit button side by side
    col1, col2 = st.columns([3, 1])
    with col1:
        model_choice = st.selectbox("Choose a model:", ["GPT-4", "Groq LLama-3.1", "GPT-4o"], key="model_choice")
    with col2:
        st.button("Send", on_click=submit)
