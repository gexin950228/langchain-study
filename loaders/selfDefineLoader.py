from typing import List, Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class CustomTextLoader(BaseLoader):
    """自定义文本文件加载器示例"""
    
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        """
        初始化自定义加载器
        
        Args:
            file_path: 要加载的文件路径
            encoding: 文件编码，默认 utf-8
        """
        self.file_path = file_path
        self.encoding = encoding
    
    def lazy_load(self) -> Iterator[Document]:
        """懒加载文档（核心方法）"""
        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                # 按段落读取（以空行分隔）
                content = []
                line_number = 0
                
                for line in f:
                    line_number += 1
                    if line.strip() == "":
                        # 遇到空行，将当前段落作为一个 Document
                        if content:
                            yield Document(
                                page_content="\n".join(content),
                                metadata={
                                    "source": self.file_path,
                                    "line_start": line_number - len(content),
                                    "line_end": line_number - 1
                                }
                            )
                            content = []
                    else:
                        content.append(line.rstrip("\n"))
                
                # 处理最后一段
                if content:
                    yield Document(
                        page_content="\n".join(content),
                        metadata={
                            "source": self.file_path,
                            "line_start": line_number - len(content) + 1,
                            "line_end": line_number
                        }
                    )
        except Exception as e:
            raise RuntimeError(f"加载文件失败: {e}")
    
    def load(self) -> List[Document]:
        """立即加载所有文档"""
        return list(self.lazy_load())


# 使用示例
if __name__ == "__main__":
    # 创建自定义加载器实例
    loader = CustomTextLoader("files/sys-error.2026-05-31.log", encoding="utf-8")
    
    # 方式1：懒加载
    print("=== 懒加载模式 ===")
    for doc in loader.lazy_load():
        print(f"段落位置: 行 {doc.metadata['line_start']}-{doc.metadata['line_end']}")
        print(f"内容预览: {doc.page_content[100:]}...")
        print("-" * 50)
    
    # 方式2：立即加载
    print("\n=== 立即加载模式 ===")
    docs = loader.load()
    print(f"共加载 {len(docs)} 个段落")
