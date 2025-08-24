src/
│
├── rag/               # 核心模块代码包
│   ├── __init__.py
│   ├── config.py                 # 模型与系统配置
│   ├── ingest.py                 # 解析 PDF 提取文本、表格、图片
│   ├── embed.py                  # 向量化文本与图像（CLIP）
│   ├── index.py                  # 构建 FAISS 向量索引 + BM25
│   ├── retrieve.py               # 多通道检索逻辑 + RRF 融合
│   ├── generate.py               # 基于上下文通过 Ollama LLM 生成回答
│   └── cli.py                    # 命令行入口
│
├── workdir/                      # 存储处理结果
│   ├── corpus.jsonl              # 所有 chunk 的元数据与内容
│   ├── images/                   # 提取出的图片
│   ├── text_embs.npy             # 文本向量
│   ├── image_embs.npy            # 图像向量
│   └── faiss_text.index          # 向量索引文件（文本）
│   └── faiss_image.index         # 向量索引文件（图像）
│
├── requirements.txt             
└── README.md                     
