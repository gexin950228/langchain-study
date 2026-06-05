from langchain_text_splitters import MarkdownHeaderTextSplitter

# 定义按标题切割的规则
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4")
]

# 加载 Markdown 文档
with open("files/大模型文档.md", "r", encoding="utf-8") as f:
    md_content = f.read()

# 创建 MarkdownHeaderTextSplitter 实例，指定按标题切割
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False  # 保留标题在内容中
)

# 切割文档
docs = markdown_splitter.split_text(md_content)

# 打印切割结果
print(f"共切割成 {len(docs)} 个文档块")
print("=" * 50)
for i, doc in enumerate(docs):
    print(f"\n--- 文档块 {i+1} ---")
    print(f"元数据: {doc.metadata}")
    print(f"内容:\n{doc.page_content}" if len(doc.page_content) > 300 else f"内容:\n{doc.page_content}")