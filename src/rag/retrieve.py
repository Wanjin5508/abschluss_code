import os
import json
import numpy as np
import faiss
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import regex as re
from config import models, settings

def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> List[str]:
    return re.findall(r"\p{L}+", text.lower())

def load_corpus(workdir: str) -> Dict[str, Dict]:
    """加载 corpus.jsonl 中的全部 chunk，按 id 映射"""
    path = os.path.join(workdir, "corpus.jsonl")
    corpus = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            ch = json.loads(line)
            corpus[ch["id"]] = ch
    return corpus


def cosine_search(index_path: str, query_vec: np.ndarray, top_k: int) -> Tuple[List[int], List[float]]:
    if not os.path.exists(index_path):
        return [], []

    if query_vec.ndim != 2 or query_vec.shape[0] != 1:
        print("Query vector shape invalid:", query_vec.shape)
        return [], []

    try:
        index = faiss.read_index(index_path)
        D, I = index.search(query_vec.astype("float32"), top_k)
        return I[0].tolist(), D[0].tolist()
    except Exception as e:
        print("FAISS error:", e)
        return [], []



def reciprocal_rank_fusion(results: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    """融合多个排序列表"""
    score_map = {}
    for res in results:
        for rank, (doc_id, _) in enumerate(res):
            score_map[doc_id] = score_map.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)

def retrieve(query: str, workdir: str, top_k: int = 15) -> List[Dict]:
    corpus = load_corpus(workdir)

    # 加载模型
    text_model = SentenceTransformer(models.text_embedding)
    clip_model = SentenceTransformer(models.clip_embedding)

    results_all = []

    # 1 文本嵌入检索
    text_ids = load_json(os.path.join(workdir, "text_ids.json"), [])
    if text_ids:
        query_vec = text_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        I, D = cosine_search(os.path.join(workdir, "faiss_text.index"), query_vec, top_k)
        results = [(text_ids[i], D[j]) for j, i in enumerate(I) if i < len(text_ids)]
        results_all.append(results)

    # 2 图像向量检索（查询文本 → CLIP 向量 → 检索图像向量）
    image_ids = load_json(os.path.join(workdir, "image_ids.json"), [])
    if image_ids:
        query_vec_img = clip_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        I2, D2 = cosine_search(os.path.join(workdir, "faiss_image.index"), query_vec_img, top_k)
        results_img = [(image_ids[i], D2[j]) for j, i in enumerate(I2) if i < len(image_ids)]
        results_all.append(results_img)

    # 3️ BM25 关键词检索
    bm25_tokens = load_json(os.path.join(workdir, "bm25_tokens.json"), [])
    bm25_ids = load_json(os.path.join(workdir, "bm25_ids.json"), [])
    if bm25_tokens and bm25_ids:
        bm25 = BM25Okapi(bm25_tokens)
        scores = bm25.get_scores(tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        results_bm25 = [(bm25_ids[i], scores[i]) for i in top_idx]
        results_all.append(results_bm25)

    # 融合排序（使用 RRF）
    fused = reciprocal_rank_fusion(results_all)
    top_ids = [doc_id for doc_id, _ in fused[:top_k]]

    # 返回对应 chunk
    return [corpus[doc_id] for doc_id in top_ids if doc_id in corpus]

def build_context(chunks: List[Dict], max_chars: int = 30000) -> str:
    blocks = []
    total = 0
    for ch in chunks:
        # label = f"[{ch['type']}] (Page {ch.get('page', '?')})"
        label = f"[Page {ch.get('page', '?')}] ({ch['type']})"
        content = ch.get("content", "").strip()
        if ch["type"] == "image":
            img_path = ch.get("meta", {}).get("image_path", "")
            content = f"[Image: {os.path.basename(img_path)}]\n{content}"
        block = f"{label}\n{content}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)

if __name__ == "__main__":
    query = "The test is cancelled with which error message?"  # page 69
    workdir = "workdir"

    results = retrieve(query, workdir)
    print(f"\n[OK] Retrieved top {len(results)} chunks:")
    for i, ch in enumerate(results, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Type: {ch['type']}, Page: {ch.get('page')}")
        print(ch.get("content")[:300], "...")
    
    context = build_context(results)
    print("\n\n[Assembled Context for LLM]:\n")
    print(context[:1000], "...")

