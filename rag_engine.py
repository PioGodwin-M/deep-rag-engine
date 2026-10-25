import re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

class DeepRAGEngine:
    def __init__(self):
        # Load models once on initialization
        self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.parent_store = {}
        self.child_chunks = []
        self.faiss_index = None
        self.bm25_index = None
        self._initialize_default_corpus()

    def _initialize_default_corpus(self):
        # Default technical corpus from our Colab test
        sections = [
            {
                "page": 1,
                "text": "Authentication & Login Troubleshooting: When users experience general login issues or forgotten credentials, direct them to the SSO self-service portal. Password resets require multi-factor verification via the mobile authenticator app. Sessions expire after 12 hours of inactivity."
            },
            {
                "page": 4,
                "text": "Gateway Error Codes & Payments: If the payment microservice throws Error 0x80070005 during Stripe webhook execution, it indicates an IAM permission denial on the Redis idempotency lock. To fix this, rotate the service role ARN and flush the stale lock key using redis-cli."
            },
            {
                "page": 9,
                "text": "Database Connection Pooling: For high-throughput FastAPI deployments, configure SQLAlchemy with pool_size=20 and max_overflow=10. If connection timeout exceptions occur under heavy load, enable PgBouncer in transaction pooling mode to prevent PostgreSQL max_connections exhaustion."
            }
        ]
        self.ingest_documents(sections)

    def ingest_documents(self, sections):
        self.parent_store = {}
        self.child_chunks = []
        
        for idx, sec in enumerate(sections):
            parent_id = f"parent_{idx}"
            self.parent_store[parent_id] = {"page": sec["page"], "full_text": sec["text"]}
            
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', sec["text"]) if s.strip()]
            for s_idx, sentence in enumerate(sentences):
                self.child_chunks.append({
                    "child_id": f"{parent_id}_c{s_idx}",
                    "parent_id": parent_id,
                    "page": sec["page"],
                    "child_text": sentence
                })
        
        # Build FAISS & BM25 indexes
        child_texts = [c["child_text"] for c in self.child_chunks]
        embeddings = self.bi_encoder.encode(child_texts, convert_to_numpy=True, normalize_embeddings=True)
        
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        
        tokenized_corpus = [re.findall(r'\w+', text.lower()) for text in child_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def hybrid_rrf_search(self, query, top_k=4, rrf_k=60):
        query_vec = self.bi_encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        faiss_scores, faiss_indices = self.faiss_index.search(query_vec, len(self.child_chunks))
        faiss_ranking = faiss_indices[0].tolist()
        
        tokenized_query = re.findall(r'\w+', query.lower())
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_ranking = np.argsort(bm25_scores)[::-1].tolist()
        
        rrf_scores = {}
        for doc_idx in range(len(self.child_chunks)):
            rank_in_faiss = faiss_ranking.index(doc_idx) + 1
            rank_in_bm25 = bm25_ranking.index(doc_idx) + 1 if bm25_scores[doc_idx] > 0 else 999
            
            faiss_contrib = 1.0 / (rrf_k + rank_in_faiss)
            bm25_contrib = (1.0 / (rrf_k + rank_in_bm25)) if bm25_scores[doc_idx] > 0 else 0.0
            
            rrf_scores[doc_idx] = {
                "rrf_score": faiss_contrib + bm25_contrib,
                "faiss_rank": rank_in_faiss,
                "bm25_rank": rank_in_bm25 if bm25_scores[doc_idx] > 0 else None
            }
            
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
        return [ {**self.child_chunks[idx], **metrics} for idx, metrics in sorted_docs[:top_k] ]

    def rerank_and_fetch_parents(self, query, candidates, top_n_parents=1):
        pairs = [[query, c["child_text"]] for c in candidates]
        cross_scores = self.reranker.predict(pairs)
        
        for idx, score in enumerate(cross_scores):
            candidates[idx]["reranker_score"] = float(score)
            
        reranked = sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)
        
        final_parents, seen = [], set()
        for item in reranked:
            pid = item["parent_id"]
            if pid not in seen:
                seen.add(pid)
                pdata = self.parent_store[pid]
                final_parents.append({
                    "parent_id": pid,
                    "page": pdata["page"],
                    "matched_child": item["child_text"],
                    "reranker_score": item["reranker_score"],
                    "full_parent_context_for_llm": pdata["full_text"]
                })
            if len(final_parents) >= top_n_parents:
                break
        return final_parents

    def verify_faithfulness(self, response_text, retrieved_parents):
        claims = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response_text) if s.strip()]
        report, supported_count = [], 0
        
        for claim in claims:
            pairs = [[claim, p["full_parent_context_for_llm"]] for p in retrieved_parents]
            scores = self.reranker.predict(pairs)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            
            is_grounded = best_score > 0.0
            if is_grounded: supported_count += 1
            
            report.append({
                "claim": claim,
                "is_grounded": is_grounded,
                "score": best_score,
                "citation": f"Page {retrieved_parents[best_idx]['page']}" if is_grounded else "UNVERIFIED"
            })
        score = supported_count / len(claims) if claims else 1.0
        return score, report