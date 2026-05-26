import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.title("AI Article Summarizer")

@st.cache_resource
def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
    # AutoTokenizer aur AutoModel ka use kar rahe hain
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

text = st.text_area("Paste your article here:", height=200)

if st.button("Summarize"):
    if text:
        with st.spinner("Summarizing..."):
            # Model ke liye inputs prepare karo
            inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
            
            # Model se summary generate karo
            summary_ids = model.generate(
                inputs, 
                max_length=150, 
                min_length=30, 
                length_penalty=2.0, 
                num_beams=4, 
                early_stopping=True
            )
            
            # Result decode karo
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.write("### Summary:")
            st.write(summary)
    else:
        st.warning("Please enter some text!")
