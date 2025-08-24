import os
import json
import numpy as np
from typing import List
from rank_bm25 import BM25Okapi
import regex as re
import faiss


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def tokenize(text: str) -> List[str]:
    """对文本进行 BM25 用的 token 分词）"""
    return re.findall(r"\p{L}+", text.lower())


def build_indices(workdir: str, use_bm25: bool = True):
    # 读取文本向量、图像向量
    text_embs_path = os.path.join(workdir, "text_embs.npy")
    image_embs_path = os.path.join(workdir, "image_embs.npy")

    text_ids_path = os.path.join(workdir, "text_ids.json")
    image_ids_path = os.path.join(workdir, "image_ids.json")

    text_embs = np.load(text_embs_path) if os.path.exists(text_embs_path) else np.zeros((0, 384), dtype=np.float32)
    image_embs = np.load(image_embs_path) if os.path.exists(image_embs_path) else np.zeros((0, 512), dtype=np.float32)

    text_ids = json.load(open(text_ids_path, encoding="utf-8")) if os.path.exists(text_ids_path) else []
    image_ids = json.load(open(image_ids_path, encoding="utf-8")) if os.path.exists(image_ids_path) else []

    # 构建文本向量索引
    if text_embs.shape[0] > 0:
        dim = text_embs.shape[1]
        index = faiss.IndexFlatIP(dim)  # 内积用于余弦相似度（前提：向量已归一化）
        index.add(text_embs)
        faiss.write_index(index, os.path.join(workdir, "faiss_text.index"))

    # 构建图像向量索引
    if image_embs.shape[0] > 0:
        dim = image_embs.shape[1]
        index_img = faiss.IndexFlatIP(dim)
        index_img.add(image_embs)
        faiss.write_index(index_img, os.path.join(workdir, "faiss_image.index"))

    # 构建 BM25 索引
    if use_bm25:
        bm25_tokens = []
        bm25_ids = []

        corpus_path = os.path.join(workdir, "corpus.jsonl")
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                text = chunk.get("content") or ""
                bm25_tokens.append(tokenize(text))
                bm25_ids.append(chunk["id"])

        bm25 = BM25Okapi(bm25_tokens)

        # 保存 BM25 所需数据（供后续恢复）
        save_json(bm25_tokens, os.path.join(workdir, "bm25_tokens.json"))
        save_json(bm25_ids, os.path.join(workdir, "bm25_ids.json"))

    print(f"索引构建完成：文本={len(text_ids)}，图像={len(image_ids)}，BM25={len(bm25_ids) if use_bm25 else 0}")

if __name__ == "__main__":
    workdir = "workdir"
    build_indices(workdir, use_bm25=True)
