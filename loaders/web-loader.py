import bs4
import asyncio
from tqdm import tqdm
from langchain_community.document_loaders import WebBaseLoader

page_url = "https://docs.langchain.com/oss/python/langchain/short-term-memory"

# 使用 bs4 过滤 HTML 标签

docs = []

async def load_documents():
    loader = WebBaseLoader(
        web_paths=[page_url],
        bs_kwargs={"parse_only": bs4.SoupStrainer(class_=lambda x: x != "navbar" and x != "footer")}
    )
    
    # 使用 tqdm 显示加载进度
    docs_list = list(loader.lazy_load())
    total_docs = len(docs_list)
    
    with tqdm(total=total_docs, desc="加载网页", unit="页") as pbar:
        for doc in docs_list:
            docs.append(doc)
            pbar.update(1)
    
    return docs

def clean_html_content(html_content: str) -> str:
    """使用 bs4 清理 HTML 内容，提取纯文本"""
    soup = bs4.BeautifulSoup(html_content, "html.parser")
    
    # 移除不需要的标签
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    
    # 提取纯文本
    text = soup.get_text(strip=True, separator="\n")
    
    # 清理多余的空白
    cleaned_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return cleaned_text

if __name__ == "__main__":
    docs = asyncio.run(load_documents())
    doc = docs[0]
    print(f"元数据: {doc.metadata}\n")
    
    # 清理 HTML 内容
    cleaned_content = clean_html_content(doc.page_content)
    print(f"清理后的内容:\n{cleaned_content[:500].strip()}")