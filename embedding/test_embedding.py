import os
import re
import warnings
from pypinyin import lazy_pinyin
# 忽略langchain-community过时警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings # type: ignore
from langchain_community.document_loaders import Docx2txtLoader # pyright: ignore[reportMissingImports]
from langchain_experimental.text_splitter import SemanticChunker # pyright: ignore[reportMissingImports]
from langchain_core.messages import HumanMessage, SystemMessage # pyright: ignore[reportMissingImports]
from langchain_openai import ChatOpenAI # type: ignore

# 导入 utils 模块中的 Milvus 工具函数
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.milvus_utils import save_to_milvus, search_milvus
# 设置智谱API密钥
os.environ["ZHIPUAI_API_KEY"] = "69695283f7034931b87220e76ef4f6f4.m11eKchDK9Ac7ZIe"
# 智谱API限制：每次最多处理64条
BATCH_SIZE = 64
# 批量获取文本向量
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
# 使用 utils/milvus_utils 中的 save_to_milvus 函数（已通过 import 导入）

# 创建向量检索函数（使用 utils/milvus_utils 中的 search_milvus 函数）
def search_project_contracts(query: str, collection_name: str = "embedding_demo", limit: int = 5):
    """检索项目组合同相关的信息（独立函数）"""
    # 创建embedding对象
    embedding = ZhipuAIEmbeddings(model="text_embedding")

    # 获取查询向量
    try:
        query_embedding = embedding.embed_query(query)
    except Exception as e:
        print(f"向量化查询失败: {e}")
        return
    
    # 使用 utils/milvus_utils 中的 search_milvus 函数搜索
    results = search_milvus(query_embedding, collection_name=collection_name, limit=limit)
    
    # 收集检索结果作为上下文
    print(f"查询关键词: '{query}'")
    print(f"找到 {len(results)} 条相关结果")
    
    context_texts = []
    for result in results:
        text = result["text"]
        context_texts.append(text)    
    
    # 将检索结果和问题发送给大模型
    context = "\n\n".join(context_texts)
    
    llm = ChatOpenAI(
        model="GLM-5V-Turbo",
        api_key=os.environ["ZHIPUAI_API_KEY"],
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    
    messages = [
        SystemMessage(content="你是一个专业的问答助手，请基于提供的上下文信息准确回答问题。如果上下文信息不足以回答问题，请说明无法回答。"),
        HumanMessage(content=f"上下文信息：\n{context}\n\n问题：{query}")
    ]
    
    response = llm.invoke(messages)
    print(f"\n === GLM大模型回答 ===")
    print(f"大模型回答: {response.content}")
    
if __name__ == '__main__':
    pwd = os.getcwd()
    filepath = os.path.join(pwd, "files\\集团智慧双碳管理系统_系统实施安装手册.docx")

    if os.path.exists(filepath):
        print(f"文件路径: {filepath}")
        
        # 从文件名提取集合名（中文转拼音）
        filename = os.path.basename(filepath)
        raw_name = os.path.splitext(filename)[0]
        pinyin_parts = lazy_pinyin(raw_name)
        pinyin_name = '_'.join(pinyin_parts).lower()
        collection_name = re.sub(r'[^a-z0-9_]', '_', pinyin_name)
        print(f"Milvus集合名: {collection_name} (源自: {raw_name})")
        
        # 读取 docx 文件内容
        loader = Docx2txtLoader(filepath)
        docs = loader.load()
        content = docs[0].page_content
        
        print(f"文件内容长度: {len(content)} 字符")
        
        # 使用语义切割
        print("\n正在进行语义切割...")
        embedding = ZhipuAIEmbeddings(model="text_embedding")
        text_splitter = SemanticChunker(embeddings=embedding)
        split_docs = text_splitter.create_documents([content])
        paragraphs = [doc.page_content for doc in split_docs]
        print(f"语义切割为 {len(paragraphs)} 个块")
        
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
        print("\n正在将向量存入 Milvus...")
        save_to_milvus(paragraphs, embeddings, collection_name=collection_name)
        print("\n数据已成功存入 Milvus!")
        
        # 调用检索函数搜索项目组联系人信息
        query = input("\n请输入您的问题: ").strip()
        if not query:
            query = "JDK怎么安装"
            print(f"未输入问题，使用默认查询: {query}")
        search_project_contracts(query, collection_name=collection_name, limit=5)
    else:
        print(f"文件不存在: {filepath}")