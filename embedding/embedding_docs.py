import os
import re
import warnings
from typing import List, Dict, Any

# 忽略 langchain-community 过时警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
<<<<<<< HEAD
from zhipuai import ZhipuAI

# 导入 utils 模块中的 Milvus 工具函数
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.milvus_utils import save_to_milvus, search_milvus

=======
from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client.index import IndexParams
from zhipuai import ZhipuAI

>>>>>>> f9ba5d08b4c35630a8f536eeb8f2032fdf1229fe
# 设置智谱API密钥
os.environ["ZHIPUAI_API_KEY"] = "69695283f7034931b87220e76ef4f6f4.m11eKchDK9Ac7ZIe"

# 智谱API限制：每次最多处理64条
BATCH_SIZE = 64

def get_batch_embeddings(texts: List[str]) -> List[List[float]]:
    """批量获取文本向量（分批处理）"""
    embedding = ZhipuAIEmbeddings(model="text_embedding")
    all_vectors = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        print(f"处理第 {i//BATCH_SIZE + 1} 批，共 {len(batch)} 条")
        vectors = embedding.embed_documents(batch)
        all_vectors.extend(vectors)
    
    return all_vectors

def load_docx_file(filepath: str) -> str:
    """加载并读取 .docx 文件内容"""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("请先安装 python-docx: pip install python-docx")
    
    doc = Document(filepath)
    full_text = []
    
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            full_text.append(paragraph.text)
    
    return '\n'.join(full_text)

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """将文本分块处理"""
    chunks = []
    text_length = len(text)
    
    if text_length <= chunk_size:
        return [text]
    
    start = 0
    while start < text_length:
        end = start + chunk_size
        # 在句子边界处分割
        if end < text_length:
            # 找最近的句号、换行或空格
            for i in range(min(chunk_overlap, chunk_size)):
                pos = end - i
                if pos > start and text[pos] in '.!?\n。！？':
                    end = pos + 1
                    break
        
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap
    
    return chunks

def generate_collection_name(text_content: str, max_length: int = 30) -> str:
    """根据文本内容生成集合名称（使用文本前缀）"""
    # 提取文本前几个词作为前缀
    words = text_content.split()[:3]  # 取前3个词
    prefix = "_".join(words).lower()
    
    # 移除非法字符，只保留字母、数字和下划线
    prefix = re.sub(r'[^a-z0-9_]', '', prefix)
    
    # 确保以字母开头
    if not prefix or not prefix[0].isalpha():
        prefix = "doc_" + prefix
    
    # 限制长度
    prefix = prefix[:max_length]
    
    # 添加固定后缀
    return f"{prefix}_embeddings"

<<<<<<< HEAD
# 使用 utils/milvus_utils 中的 save_to_milvus 函数（已通过 import 导入）

def search_documents(query: str, collection_name: str = "embedding_docx", limit: int = 5) -> List[Dict[str, Any]]:
    """从 Milvus 中检索相关文档，返回检索结果列表（使用 utils/milvus_utils）"""
=======
def save_to_milvus(texts: List[str], embeddings: List[List[float]], collection_name: str = "embedding_docx") -> bool:
    """将文本和向量存入 Milvus"""
    try:
        # 连接 Milvus
        client = MilvusClient(
            uri="http://172.18.83.231:19530",
            user="root",
            password="ufa4A$hiTyTeP@V$a"
        )
        print("成功连接到 Milvus!")
        
        # 检查集合是否存在，如果存在则删除
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            print(f"已删除已存在的集合: {collection_name}")
        
        # 创建包含 text 字段的集合
        schema = client.create_schema(auto_id=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=len(embeddings[0]))
        
        client.create_collection(collection_name=collection_name, schema=schema)
        print(f"已创建集合: {collection_name}")
        
        # 准备数据
        data = [
            {
                "id": i,
                "text": texts[i],
                "vector": embeddings[i]
            }
            for i in range(len(texts))
        ]
        
        # 插入数据
        result = client.insert(collection_name=collection_name, data=data)
        print(f"已插入 {len(data)} 条数据，插入结果: {result}")
        
        # 为向量字段创建索引
        index_params = IndexParams()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128}
        )
        
        client.create_index(
            collection_name=collection_name,
            index_params=index_params
        )
        print(f"已为向量字段创建索引")
        
        # 加载集合到内存
        client.load_collection(collection_name=collection_name)
        print(f"集合已加载到内存")
        
        # 查看集合统计信息
        stats = client.get_collection_stats(collection_name)
        print(f"集合统计信息: {stats}")
        
        return True
        
    except Exception as e:
        print(f"连接 Milvus 失败: {e}")
        return False

def search_documents(query: str, collection_name: str = "embedding_docx", limit: int = 5) -> List[Dict[str, Any]]:
    """从 Milvus 中检索相关文档，返回检索结果列表"""
>>>>>>> f9ba5d08b4c35630a8f536eeb8f2032fdf1229fe
    print(f"\n=== 检索相关文档 ===")
    
    # 创建 embedding 对象
    embedding = ZhipuAIEmbeddings(model="text_embedding")
    
    # 获取查询向量
    try:
        query_embedding = embedding.embed_query(query)
    except Exception as e:
        print(f"向量化查询失败: {e}")
        return []
    
<<<<<<< HEAD
    # 使用 utils/milvus_utils 中的 search_milvus 函数搜索
    try:
        results = search_milvus(query_embedding, collection_name=collection_name, limit=limit)
        
        # 整理结果
        retrieved_docs = []
        for i, result in enumerate(results):
            text = result["text"]
=======
    # 连接 Milvus
    client = MilvusClient(
        uri="http://172.18.83.231:19530",
        user="root",
        password="ufa4A$hiTyTeP@V$a"
    )
    
    # 搜索 Milvus
    try:
        results = client.search(
            collection_name=collection_name,
            data=[query_embedding],
            limit=limit,
            output_fields=["text"]
        )
        
        # 整理结果
        retrieved_docs = []
        for i, result in enumerate(results[0]):
            text = result["entity"].get("text", "无文本内容")
>>>>>>> f9ba5d08b4c35630a8f536eeb8f2032fdf1229fe
            distance = result["distance"]
            retrieved_docs.append({
                "similarity": 1 - distance,
                "content": text
            })
            print(f"{i+1}. 相似度: {1 - distance:.4f}")
            print(f"   内容: {text[:100]}..." if len(text) > 100 else f"   内容: {text}")
            print()
        
        return retrieved_docs
            
    except Exception as e:
        print(f"检索失败: {e}")
        return []

def call_glm_model(query: str, context_docs: List[Dict[str, Any]]) -> str:
    """调用智谱 GLM-5.1 大模型，基于检索结果生成答案"""
    print("\n=== 调用 GLM-5.1 大模型 ===")
    
    # 构建提示词
    context = "\n\n".join([f"文档{i+1}（相似度: {doc['similarity']:.4f}）:\n{doc['content']}" 
                          for i, doc in enumerate(context_docs)])
    
    prompt = f"""
请根据以下提供的上下文信息，回答用户的问题。

上下文信息：
{context}

用户问题：
{query}

请基于上下文信息给出准确的回答。如果上下文信息不足以回答问题，请说明无法回答。
"""
    
    try:
        # 初始化智谱客户端
        client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])
        
        # 调用 GLM-5.1 模型
        response = client.chat.completions.create(
            model="glm-5.1",
            messages=[
                {"role": "system", "content": "你是一个专业的问答助手，请基于提供的上下文信息回答问题。"},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        
        answer = response.choices[0].message.content
        print(f"GLM-5.1 回答:\n{answer}")
        return answer
        
    except Exception as e:
        print(f"调用 GLM-5.1 失败: {e}")
        return ""

def rag_query(query: str, collection_name: str = "embedding_docx", limit: int = 5) -> str:
    """完整的 RAG 查询流程：检索 + 大模型回答"""
    # 1. 检索相关文档
    retrieved_docs = search_documents(query, collection_name, limit)
    
    if not retrieved_docs:
        print("未检索到相关文档")
        return ""
    
    # 2. 调用大模型生成答案
    answer = call_glm_model(query, retrieved_docs)
    
    return answer

def process_docx_to_milvus(docx_path: str) -> bool:
    """处理 .docx 文件并将向量化结果存入 Milvus"""
    print(f"开始处理 .docx 文件: {docx_path}")
    
    # 加载文档内容
    content = load_docx_file(docx_path)
    print(f"文档内容长度: {len(content)} 字符")
    
    # 根据文本内容生成集合名称
    collection_name = generate_collection_name(content)
    print(f"生成的集合名称: {collection_name}")
    
    # 分块处理
    chunks = chunk_text(content)
    print(f"分割为 {len(chunks)} 个文本块")
    
    # 向量化
    print("\n正在进行向量化...")
    embeddings = get_batch_embeddings(chunks)
    
    print(f"\n向量化完成!")
    print(f"生成了 {len(embeddings)} 个向量")
    print(f"每个向量维度: {len(embeddings[0])}")
    
    # 存入 Milvus
    print("\n正在将向量存入 Milvus...")
    success = save_to_milvus(chunks, embeddings, collection_name)
    
    if success:
        print("\n数据已成功存入 Milvus!")
    
    return success

if __name__ == "__main__":
    pwd = os.getcwd()
    docx_path = os.path.join(pwd, "files", "集团智慧双碳管理系统_运维手册.docx")  # 替换为您的 .docx 文件路径
    
    if os.path.exists(docx_path):
        # 处理文档并存入 Milvus
        success = process_docx_to_milvus(docx_path)
        
        if success:
            # 获取集合名称用于检索
            content = load_docx_file(docx_path)
            collection_name = generate_collection_name(content)
            
            # 示例 RAG 查询
            print("\n" + "="*50)
            print("示例 RAG 查询")
            print("="*50)
            query = input("请输入您的问题: ")
            rag_query(query, collection_name)
    else:
        print(f"文件不存在: {docx_path}")
        print("请将 .docx 文件放入 files 目录，并修改 docx_path 变量")