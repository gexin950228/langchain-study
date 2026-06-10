import base64
import os
import io
import fitz
from PIL import Image
from pathlib import Path
# Pillow替代PIL，但是导入还是PIL
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def pdf_to_base64(pdf_path: str, page_number: int) -> str:
    current_dir = os.getcwd()
    abs_path = os.path.join(current_dir, pdf_path)
    print("PDF路径:", abs_path)
    
    # 确定正确的PDF路径
    if os.path.exists(pdf_path):
        file_path = pdf_path
    elif os.path.exists(abs_path):
        file_path = abs_path
    else:
        raise FileNotFoundError(f"PDF文件未找到: {pdf_path} 或 {abs_path}")
    
    # 处理PDF并转换为base64
    pdf_document = fitz.open(file_path)
    page = pdf_document.load_page(page_number-1)
    pix = page.get_pixmap()
    
    # 使用 frombuffer 替代 frombytes，避免 decoder 问题
    img = Image.frombuffer("RGB", (pix.width, pix.height), pix.samples, "raw", "RGB", 0, 1)
    
    # 将图片保存到本地以便查看
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"page_{page_number}.png")
    img.save(output_path, format="PNG")
    print(f"图片已保存到: {output_path}")
    
    # 同时生成base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_str
    
if __name__ == "__main__":
    pdf_path = "files/2021年中国Z世代手办消费趋势研究报告.pdf"
    page_number = 11
    base64_str = pdf_to_base64(pdf_path, page_number)
    print("base64_str: ",base64_str)
    llm = ChatOpenAI(
        api_key="69695283f7034931b87220e76ef4f6f4.m11eKchDK9Ac7ZIe",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        temperature=0,
        model = "glm-4v",
    )
    query = "新一线城市消费者占比有多少？"
    message = HumanMessage(
        content=[
            {"type": "text", "text": query},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_str}"
                }
            }
        ]
    )
    response = llm.invoke([message])
    print(response.content)