from langchain_community.document_loaders import PyPDFLoader
import os
import asyncio 

pwd = os.getcwd()
filepath_path = os.path.join(pwd, "files/CDN技术详解.pdf")
print(filepath_path)


async def load_pages(pdf_path: str):
    loader = PyPDFLoader(file_path=pdf_path)
    # Use aload() instead of alazy() - alazy() doesn't exist
    pages = await loader.aload()
    return pages

if __name__ == "__main__":
    pages = asyncio.run(load_pages(filepath_path))
    print(pages[0].page_content)