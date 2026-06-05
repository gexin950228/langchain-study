import base64
import io
import os

import fitz
from PIL import Image
from IPython.display import Image as IPImage
from IPython.display import display
from zhipuai import ZhipuAI

def pdf_page_to_base64(pdf_path: str, page_num: int) -> str:
    print(f"尝试打开PDF文件: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"文件不存在: {pdf_path}")
        return ""
    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(page_num-1)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_str
    except Exception as e:
        print(f"发生错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return ""


client = ZhipuAI(api_key="69695283f7034931b87220e76ef4f6f4.m11eKchDK9Ac7ZIe", base_url="https://open.bigmodel.cn/api/paas/v4")

if __name__ == "__main__":
    print("开始执行脚本...")
    pdf_path = "files/2021年中国Z世代手办消费趋势研究报告.pdf"
    print(f"PDF路径: {pdf_path}")
    base64_img = pdf_page_to_base64(pdf_path, 11)
    display(IPImage(data=base64.b64decode(base64_img)))
    query = "新一线城市消费者占比有多少？"
    # 使用zhipuai库的正确格式调用多模态API
    try:
        response = client.chat.completions.create(
            model="glm-5",
            messages=[
                {
                    "role": "user",
                    "content": f"{query}\n\n![image](data:image/png;base64,{base64_img})"
                }
            ]
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"API调用错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()