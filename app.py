import streamlit as st
from rag_engine import DeepRAGEngine

st.set_page_config(page_title="Deep RAG & NLI Auditor", layout="wide")

@st.cache_resource
def load_engine():
    return DeepRAGEngine()

engine = load_engine()

st.title("🔍 Production Deep RAG Engine")
st.markdown("Demonstrating **Hybrid Search (BM25 + FAISS)**, **Reciprocal Rank Fusion**, **Cross-Encoder Reranking**, and **NLI Faithfulness Auditing**.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Query & Retrieval Test")
    query = st.text_input("Enter technical query:", "How do we resolve the 0x80070005 failure?")
    
    if st.button("Run Hybrid Search & Rerank"):
        with st.spinner("Running retrieval pipeline..."):
            candidates = engine.hybrid_rrf_search(query, top_k=4)
            final_parents = engine.rerank_and_fetch_parents(query, candidates, top_n_parents=1)
            
            st.session_state['last_query'] = query
            st.session_state['last_parents'] = final_parents
            st.session_state['last_candidates'] = candidates

    if 'last_parents' in st.session_state:
        st.markdown("### Top Retrieved Parent Context")
        p = st.session_state['last_parents'][0]
        st.info(f"**Parent ID:** {p['parent_id']} (Page {p['page']}) | **Score:** {p['reranker_score']:.4f}\n\n{p['full_parent_context_for_llm']}")

with col2:
    st.subheader("2. Hallucination & Faithfulness Audit")
    default_response = (
        "To fix Error 0x80070005, rotate the IAM service role ARN and flush the stale Redis lock key using redis-cli. "
        "Additionally, you must restart the Kubernetes ingress cluster every 24 hours to clear cache."
    )
    llm_response = st.text_area("Simulated LLM Response to Audit:", default_response, height=120)
    
    if st.button("Audit Faithfulness"):
        if 'last_parents' not in st.session_state:
            st.warning("Please run retrieval first!")
        else:
            score, report = engine.verify_faithfulness(llm_response, st.session_state['last_parents'])
            st.metric(label="Overall Faithfulness Score", value=f"{score*100:.1f}%")
            
            for item in report:
                if item['is_grounded']:
                    st.success(f"✅ **Grounded** (Score: {item['score']:.2f}) [{item['citation']}]\n\n> {item['claim']}")
                else:
                    st.error(f"🚨 **Hallucination Flagged** (Score: {item['score']:.2f})\n\n> {item['claim']}")