
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="DSA Interview Assistant",
    page_icon="🤖"
)

st.title("🤖 DSA Interview Assistant")

@st.cache_resource
def load_model():

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    return tokenizer, model

tokenizer, model = load_model()

prompt = st.text_area("Ask a DSA Question")

if st.button("Generate Answer"):

    if prompt.strip() == "":
        st.warning("Please enter a question.")

    else:

        formatted_prompt = f'''
### Instruction:
{prompt}

### Response:
'''

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        st.subheader("Answer")

        st.write(response)
