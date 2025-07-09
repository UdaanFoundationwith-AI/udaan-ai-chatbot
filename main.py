import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set Streamlit page configuration
st.set_page_config(page_title="Udaan AI", page_icon="🕊️")

# Custom header
st.markdown("""
    <h1 style='text-align: center; color: #8A2BE2;'>🕊️ Udaan AI</h1>
    <h3 style='text-align: center; color: #555;'>Giving Wings to Every Mind</h3>
    <hr>
""", unsafe_allow_html=True)

# Load lightweight transformer model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizer, model

tokenizer, model = load_model()

# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask something... (English only for now)")

if prompt:
    # Display user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            output = model.generate(
                input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 14px;'>
        Made with 💜 by <b>Maheen Khan</b>
    </div>
""", unsafe_allow_html=True)


