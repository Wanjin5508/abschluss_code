from retrieve import retrieve, build_context
import requests
from config import models


SYSTEM_PROMPT = """You are a product manual assistant. Only answer based on the provided manual excerpts below:
- If you cannot find an answer from the content, say: "The manual does not contain relevant information."
- When answering, cite the excerpt number like [1], [2], which corresponds to the relevant section.
- Each excerpt shows the original page number and content type (text, table, image).
- Be as accurate and detailed as possible. Include explanations or steps if relevant.
"""



def build_prompt(context: str, question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n手册摘录:\n\n{context}\n\n用户问题:\n{question}\n\n回答:"


def call_ollama(prompt: str, model: str = "qwen2.5:latest") -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
        "num_predict": 500  # 或更高，最大生成 token 数
    }
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print(f"Ollama 回答：", result.get("response", ""))
        return result.get("response", "").strip()
    except Exception as e:
        print("调用 Ollama 出错：", e)
        return "[错误] 无法生成回答"



def generate_answer(query: str, workdir: str) -> str:
    print(f"[Query] {query}")
    
    # 步骤 1：检索相关内容
    chunks = retrieve(query, workdir=workdir)
    
    # 步骤 2：拼接上下文
    context = build_context(chunks)
    
    # 步骤 3：构建 prompt
    prompt = build_prompt(context, query)
    
    # 步骤 4：调用本地 LLM（Ollama）
    answer = call_ollama(prompt, model=models.ollama_model)
    
    # 输出
    print("\n[✓] 生成回答：\n")
    print(answer)
    return answer


if __name__ == "__main__":
    test_question = "What is the reason of Error: Overvoltage?" # table, page 66
    # test_question = "What for support of partial discharge measurements can the product offer?" 
    # test_question = "What is displayed on the screen 'Temperatures'?"
    workdir = "workdir"
    generate_answer(test_question, workdir)


