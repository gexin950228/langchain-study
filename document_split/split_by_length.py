from langchain_text_splitters import CharacterTextSplitter
import asyncio
from langchain_community.document_loaders import PyPDFLoader

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=10
)

async def load_pdf(pdf_path: str):
    loader = PyPDFLoader(file_path=pdf_path)
    # Use aload() instead of alazy() - alazy() doesn't exist
    pages = await loader.aload()
    return pages

if __name__ == "__main__":
    pages = asyncio.run(load_pdf("files/8.国家能源集团司库管控系统_用户操作手册_往来客商管理.pdf"))
    docs = text_splitter.split_documents(pages)
    print(docs[19].page_content)