import streamlit as st
from generate_answer import generate_answer

st.set_page_config(page_title="arXiv Research Assistant", page_icon="📚")

st.title("📚 arXiv Research Assistant")
st.write(
    "Ask a question about machine learning or NLP research. "
    "Answers come from 2,821 recent arXiv abstracts and cite their sources."
)

question = st.text_input(
    "Your question",
    placeholder="e.g. What are recent approaches to few-shot learning?",
)

if st.button("Ask") and question.strip():
    with st.spinner("Searching papers and writing an answer..."):
        answer, papers = generate_answer(question)

    st.subheader("Answer")
    st.markdown(answer)

    st.subheader("Sources")
    for i, p in enumerate(papers, 1):
        url = f"https://arxiv.org/abs/{p['arxiv_id']}"
        with st.expander(f"[{i}] {p['title']}"):
            st.markdown(f"[View on arXiv]({url})")
            st.write(p["abstract"])
            