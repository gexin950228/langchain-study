import os
import sys
import configparser
import warnings

# 添加项目根目录到路径，以便导入 utils 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 忽略 langchain-community 过时警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client.index import IndexParams


# 导入 utils 模块中的 Milvus 工具函数
from utils.milvus_utils import save_to_milvus, search_milvus

# 从 env.ini 加载配置
def load_milvus_config(config_path: str = "env.ini") -> dict:
    """从配置文件加载 Milvus 配置"""
    config = configparser.ConfigParser()
    if os.path.exists(config_path):
        config.read(config_path, encoding='utf-8')
    else:
        # 从上级目录查找
        parent_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), "env.ini")
        if os.path.exists(parent_config):
            config.read(parent_config, encoding='utf-8')
        else:
            raise FileNotFoundError(f"配置文件 env.ini 未找到")
    
    milvus_config = {
        "host": config.get("milvus", "host", fallback="localhost"),
        "port": config.getint("milvus", "port", fallback=19530),
        "user": config.get("milvus", "access_key", fallback="").strip('"'),
        "password": config.get("milvus", "secret_key", fallback="").strip('"')
    }
    return milvus_config

# 设置智谱API密钥
os.environ["ZHIPUAI_API_KEY"] = "69695283f7034931b87220e76ef4f6f4.m11eKchDK9Ac7ZIe"
# 智谱API限制：每次最多处理64条
BATCH_SIZE = 64
def get_batch_embeddings(texts: list) -> list:
    """批量获取文本向量（分批处理）"""
    embedding = ZhipuAIEmbeddings(model="text_embedding")
    all_vectors = []
 
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        print(f"处理第 {i//BATCH_SIZE + 1} 批，共 {len(batch)} 条")
        vectors = embedding.embed_documents(batch)
        all_vectors.extend(vectors)
    
    return all_vectors


# 创建向量检索函数（使用 utils/milvus_utils 中的 search_milvus 函数）
def search_project_contacts(query: str, limit: int = 5):
    """检索项目组联系人相关的信息（独立函数）"""
    print(f"\n=== 检索项目组联系人相关信息 ===")
    
    # 创建 embedding 对象（使用正确的模型名称）
    embedding = ZhipuAIEmbeddings(model="text_embedding")
    
    # 获取查询向量
    try:
        query_embedding = embedding.embed_query(query)
    except Exception as e:
        print(f"向量化查询失败: {e}")
        return
    
    # 使用 utils/milvus_utils 中的 search_milvus 函数搜索
    results = search_milvus(query_embedding, collection_name="embedding_demo", limit=limit)
    
    # 收集检索结果作为上下文
    print(f"查询关键词: '{query}'")
    print(f"找到 {len(results)} 条相关结果:\n")
    
    context_texts = []
    for i, result in enumerate(results):
        text = result["text"]
        distance = result["distance"]
        context_texts.append(text)
        print(f"{i+1}. 相似度: {1 - distance:.4f}")
        print(f"   内容: {text}")
        print()
    
    # 将检索结果和问题发送给大模型
    context = "\n\n".join(context_texts)
    
    llm = ChatOpenAI(
        model="GLM-5V-Turbo",
        api_key=os.environ["ZHIPUAI_API_KEY"],
        base_url="https://open.bigmodel.cn/api/paas/v4"
    )
    
    messages = [
        SystemMessage(content="你是一个专业的问答助手，请基于提供的上下文信息准确回答问题。如果上下文信息不足以回答问题，请说明无法回答。"),
        HumanMessage(content=f"上下文信息：\n{context}\n\n问题：{query}")
    ]
    
    response = llm.invoke(messages)
    print(f"\n=== GLM 大模型回答 ===")
    print(response.content)

if __name__ == "__main__":
    pwd = os.getcwd()
    filepath_path = os.path.join(pwd, "files\\集团智慧双碳管理系统_运维手册.docx")
    
    if os.path.exists(filepath_path):
        print(f"文件路径: {filepath_path}")
        
        # 读取 docx 文件内容
        loader = Docx2txtLoader(filepath_path)
        docs = loader.load()
        content = docs[0].page_content
        embedding = ZhipuAIEmbeddings(model="text_embedding")
        text_splitter = SemanticChunker(embeddings=embedding)
        split_docs = text_splitter.create_documents([content])
        paragraphs = [doc.page_content for doc in split_docs]
        
        # 打印每个块的长度和前50个字符
        for i, p in enumerate(paragraphs):
            preview = p[:50].replace('\n', ' ')
            print(f"  块 {i+1}: 长度={len(p)}, 预览={preview}...")
        
        # 调用智谱embedding模型进行向量化
        print("\n正在进行向量化...")
        embeddings = get_batch_embeddings(paragraphs)
        
        print(f"\n向量化完成!")
        print(f"生成了 {len(embeddings)} 个向量")
        print(f"每个向量维度: {len(embeddings[0])}")
        
        # 将向量存入 Milvus
        save_to_milvus(paragraphs, embeddings)
        
        # 调用检索函数搜索项目组联系人信息
        search_project_contacts("redis服务的地址和启动方式是什么", 5)
    else:
        print(f"文件不存在: {filepath_path}")