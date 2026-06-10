from langchain_community.document_loaders import CSVLoader
import os

def load_csv(csv_file:str):
    """懒加载 CSV 文件"""
    loader = CSVLoader(file_path=csv_file, source_column="地址", encoding="GBK")
    return loader.lazy_load()

def getFullPath(csv_file:str):
    """获取 CSV 文件的完整路径"""
    pwd = os.getcwd()
    full_path = os.path.join(pwd, csv_file)
    if os.path.exists(full_path):
        return full_path
    else:
        exit(1)

if __name__ == "__main__":
    csv_file = "files/电力运营管控测试环境服务器.csv"
    full_path = getFullPath(csv_file)
    docs_generator = load_csv(full_path)
    
    # 懒加载返回生成器，可以逐个处理
    count = 0
    for doc in docs_generator:
        count += 1
        print(f"记录 {count}: {doc.page_content}")
    
    print(f"加载 {count} 条记录")