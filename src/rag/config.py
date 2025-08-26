from dataclasses import dataclass

@dataclass
class Models:
    # 文本嵌入模型（MiniLM 提供较快的文本语义检索）
    text_embedding: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384维度

    # 图像 + 文本嵌入模型（用于 CLIP 多模态图文检索）
    clip_embedding: str = "sentence-transformers/clip-ViT-B-32"     # 512维度

    # 本地 Ollama 模型名称（你使用的是 gemma3:4b）
    ollama_model: str = "qwen2.5:latest"   # "gemma3:4b"

@dataclass
class Settings:
    # 是否对图片执行 OCR（使用 pytesseract）
    do_ocr_on_images: bool = False

    # 图片最大尺寸（超出将被缩放，用于加快 CLIP 推理）
    max_image_dim: int = 1280

    # 检索候选数量（RRF 前的 top_k）
    top_k_candidates: int = 12

    # 最终用于生成上下文的 chunk 数量
    final_contexts: int = 10

    # 是否启用 BM25 检索补充
    use_bm25: bool = True

    # 生成时最大拼接的上下文字符数
    max_context_chars: int = 30000

models = Models()
settings = Settings()
