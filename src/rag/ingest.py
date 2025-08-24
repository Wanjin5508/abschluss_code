import os
import json
import io
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image

from config import settings


@dataclass
class Chunk:
    id: str
    page: int
    type: str  # "text" | "table" | "image"
    content: str  # 文本 或 表格内容，图片则为 OCR 文本（如果启用）
    meta: Dict[str, Any]

    def to_dict(self):
        return asdict(self)


def extract_text_and_tables(pdf_path: str) -> List[Chunk]:
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # 提取文本, 采用嵌套循环, 逐页提取文本和其中潜在的表格
            text = page.extract_text()
            if text:
                chunks.append(Chunk(
                    id=f"text_{i}",
                    page=i,
                    type="text",
                    content=text.strip(),
                    meta={}
                ))
            # 提取表格
            tables = page.extract_tables()
            for j, table in enumerate(tables or []):
                if not table:
                    continue
                # 转换为 markdown 表格
                header = [str(cell or "").strip() for cell in table[0]]
                rows = table[1:]

                md_lines = [
                    "| " + " | ".join(header) + " |",
                    "| " + " | ".join(["---"] * len(header)) + " |"
                ]
                for row in rows:
                    row_clean = [str(cell or "").strip() for cell in row]
                    md_lines.append("| " + " | ".join(row_clean) + " |")
                chunks.append(Chunk(
                    id=f"table_{i}_{j}",
                    page=i,
                    type="table",
                    content="\n".join(md_lines),
                    meta={}
                ))
    return chunks


def extract_images(pdf_path: str, image_dir: str) -> List[Chunk]:
    os.makedirs(image_dir, exist_ok=True)
    chunks = []
    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        page = doc[i]
        images = page.get_images(full=True)
        for j, img in enumerate(images):
            xref = img[0]
            base = doc.extract_image(xref)
            image_bytes = base["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_path = os.path.join(image_dir, f"page_{i+1}_img_{j+1}.png")
            image.save(image_path)

            # 可选 OCR
            ocr_text = ""
            if settings.do_ocr_on_images:
                try:
                    import pytesseract
                    ocr_text = pytesseract.image_to_string(image)
                except Exception:
                    pass

            chunks.append(Chunk(
                id=f"image_{i+1}_{j+1}",
                page=i + 1,
                type="image",
                content=ocr_text.strip(),
                meta={"image_path": image_path}
            ))
    return chunks


def ingest_pdf(pdf_path: str, workdir: str) -> str:
    os.makedirs(workdir, exist_ok=True)
    image_dir = os.path.join(workdir, "images")
    chunks = extract_text_and_tables(pdf_path)
    chunks += extract_images(pdf_path, image_dir)

    output_path = os.path.join(workdir, "corpus.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")

    print(f"提取完成，总共 {len(chunks)} 个 chunk，保存至: {output_path}")
    return output_path



if __name__ == "__main__":
    test_pdf_path = "../1_R5001_BA9_DE-EN_2022-05.pdf"     
    test_workdir = "workdir"

    ingest_pdf(test_pdf_path, test_workdir)

    # 可选：读取并打印前 2 个 chunk 查看内容
    corpus_path = os.path.join(test_workdir, "corpus.jsonl")
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i in range(2):
            line = f.readline()
            if not line:
                break
            print(f"\n--- Chunk {i+1} ---")
            print(json.loads(line))





