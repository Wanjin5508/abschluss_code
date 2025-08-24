import os
import json
import numpy as np
from typing import List, Dict, Any
from PIL import Image
from sentence_transformers import SentenceTransformer


def load_chunks(corpus_path: str) -> List[Dict[str, Any]]:
    """读取 ingest 阶段生成的 corpus.jsonl 文件"""
    chunks = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def save_json(obj, path):
    """保存 JSON 文件（如 id 映射）"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def embed_corpus(workdir: str, text_model_name: str, clip_model_name: str):
    # 读取已解析的 chunk（包含文本/表格/图像）
    corpus_path = os.path.join(workdir, "corpus.jsonl")
    chunks = load_chunks(corpus_path)

    # 加载两个嵌入模型
    text_model = SentenceTransformer(text_model_name)
    clip_model = SentenceTransformer(clip_model_name)

    # 分别构建待嵌入内容的列表
    text_chunks, text_ids = [], []
    image_paths, image_ids = [], []

    for ch in chunks:
        if ch["type"] in ("text", "table"):
            # 文本和表格都可以看作是文本，统一送入文本模型
            payload = ch["content"]
            if payload:
                text_chunks.append(payload)
                text_ids.append(ch["id"])

        elif ch["type"] == "image":
            # 提取图片路径，确保文件存在
            img_path = ch.get("meta", {}).get("image_path")
            if img_path and os.path.exists(img_path):
                image_paths.append(img_path)
                image_ids.append(ch["id"])

    # 嵌入文本类 chunk
    if text_chunks:
        print(f"[Text] Encoding {len(text_chunks)} 文本/表格段落...")
        text_embs = text_model.encode(text_chunks, convert_to_numpy=True, normalize_embeddings=True)
    else:
        text_embs = np.zeros((0, 384), dtype="float32")

    # 嵌入图片类 chunk
    if image_paths:
        print(f"[Image] Encoding {len(image_paths)} 图像...")
        pil_images = [Image.open(p).convert("RGB") for p in image_paths]
        image_embs = clip_model.encode(pil_images, convert_to_numpy=True, normalize_embeddings=True)
    else:
        image_embs = np.zeros((0, 512), dtype="float32")

    # 保存向量和 ID 映射
    np.save(os.path.join(workdir, "text_embs.npy"), text_embs)
    np.save(os.path.join(workdir, "image_embs.npy"), image_embs)

    save_json(text_ids, os.path.join(workdir, "text_ids.json"))
    save_json(image_ids, os.path.join(workdir, "image_ids.json"))
    save_json(image_paths, os.path.join(workdir, "image_paths.json"))

    print(f"嵌入完成: 文本 {len(text_ids)} 段，图像 {len(image_ids)} 张")


if __name__ == "__main__":
    workdir = "workdir"
    embed_corpus(
        workdir=workdir,
        text_model_name="all-MiniLM-L6-v2",
        clip_model_name="clip-ViT-B-32"
    )
