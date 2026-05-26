import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)

@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="sshleifer/distilbart-cnn-12-6"
    )

summarizer = load_model()


def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


st.title("📝 AI Text Summarizer")
st.write("Paste any article or paragraph and get an AI-generated summary.")

text = st.text_area("Enter Text", height=300, placeholder="Paste your article here...")

max_length = st.slider("Summary Length", 30, 200, 80)

if st.button("Generate Summary"):

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Generating Summary..."):

        chunks = chunk_text(text)

        summaries = []
        progress_bar = st.progress(0)

        for idx, chunk in enumerate(chunks):

            result = summarizer(
                chunk,
                max_length=max_length,
                min_length=20,
                do_sample=False
            )

            summaries.append(result[0]["generated_text"])
            progress_bar.progress((idx + 1) / len(chunks))

        combined_summary = " ".join(summaries)

        if len(combined_summary.split()) > 500:

            final_result = summarizer(
                combined_summary,
                max_length=max_length,
                min_length=20,
                do_sample=False
            )

            final_summary = final_result[0]["generated_text"]

        else:
            final_summary = combined_summary

    st.subheader("Summary")
    st.success(final_summary)

    st.subheader("Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Original Words", len(text.split()))

    with col2:
        st.metric("Summary Words", len(final_summary.split()))