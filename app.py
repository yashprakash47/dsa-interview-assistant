import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="DSA Interview Assistant",
    page_icon="🤖"
)

st.title("🤖 DSA Interview Assistant")

st.write("Ask any DSA-related question.")

@st.cache_resource
def load_model():

    generator = pipeline(
        task="text2text-generation",
        model="google/flan-t5-small"
    )

    return generator

generator = load_model()

prompt = st.text_area("Enter your question")

if st.button("Generate Answer"):

    if prompt.strip() == "":
        st.warning("Please enter a question.")

    else:

        response = generator(
            prompt,
            max_length=80
        )

        answer = response[0]["generated_text"]

        st.subheader("Answer")

        st.write(answer)
