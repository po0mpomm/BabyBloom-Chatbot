import streamlit as st
import os
from dotenv import load_dotenv
from rag_engine import BabyBloomEngine

load_dotenv()

st.title("👶 Baby Blooms: Your Newborn Care Assistant")
st.caption("Empathetic answers grounded in medical textbooks.")

if not os.getenv("GROQ_API_KEY") or not os.getenv("COHERE_API_KEY"):
    st.error("API keys for Groq and Cohere are missing from your .env file.")
    st.stop()

@st.cache_resource
def get_engine():
    try:
        return BabyBloomEngine()
    except Exception as e:
        st.error(f"Failed to initialize the chatbot. Error: {e}")
        st.stop()

engine = get_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about newborn care..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Baby Blooms is thinking..."):
            try:
                result, intent = engine.ask(prompt, st.session_state.messages)
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                st.error(f"Error: {e}")