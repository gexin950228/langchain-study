import os
import re
import warnings
# 忽略langchain-community过时警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings # type: ignore
from langchain_community.document_loaders import Docx2txtLoader # pyright: ignore[reportMissingImports]
from langchain_experimental.text_splitter import SemanticChunker # pyright: ignore[reportMissingImports]
from langchain_core.messages import HumanMessage, SystemMessage # pyright: ignore[reportMissingImports]
from pymilvus import MilvusClient, DataType # pyright: ignore[reportMissingImports]
from pymilvus.milvus_client.index import IndexParams # pyright: ignore[reportMissingImports]
from langchain_openai import ChatOpenAI # type: ignore
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
# 修改save_to_milvus函数，创建包含text字段的集合
def save_to_milvus(texts: list, embeddings: list, collection_name: str = "embedding_demo"):
    """将文本向量保存到Milvus"""
    try: 
        # 连接Milvus数据库（使用http协议，端口19530）
        client = MilvusClient(
            uri="http://172.27.176.25:19530",
            user="root",
            password="69695283f7034931b87220e76ef4f6f4.m11eKchDK9Ac7ZIe",
        )
        print("连接Milvus成功")
        
        # 检查集合是否存在,如果存在则删除
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            print(f"删除集合 {collection_name} 成功")
            
        # 创建包含text的字段集合
        schema = client.create_schema(auto_id=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=len(embeddings[0]))
        
        client.create_collection(collection_name, schema=schema)
        print(f"创建集合 {collection_name} 成功")
        
        # 准备数据（包含text字段）
        data = [
            {
                "id": i,
                "text": texts[i],
                "vector": embeddings[i],
            }
            for i in range(len(texts))
        ]
        
        # 插入数据
        result = client.insert(collection_name, data=data)
        print(f"已插入 {len(data)} 条数据（包含文本内容），插入结果: {result}")
        
        # 为向量字段创建索引（Milvus需要索引才能加载和搜索）
        index_params = IndexParams()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128}
        )
        client.create_index(
            collection_name=collection_name,
            index_params=index_params,  
        )
        
        # 加载集合到内存（Milvus需要先加载才能检索)
        client.load_collection(collection_name=collection_name  )
        print(f"集合已加载到内存")
        
        # 查看集合统计信息
        stats = client.get_collection_stats(collection_name)
        print(f"集合 {collection_name} 统计信息: {stats}")
        
    except Exception as e:
        print(f"连接Milvus失败: {e}")
        print("将使用本地检索模式")
        
#  创建向量检索函数
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
    
    # 连接Milvus
    client = MilvusClient(
        uri="http://172.27.176.25:19530",
        user="root",
        password="ufa4A$hiTyTeP@V$a"
    )
    
    # 搜索Milvus
    results = client.search(
        collection_name=collection_name,
        data=[query_embedding],
        limit=limit,
        output_fields=["text"],
        anns_field="vector",
        search_params={"nprobe": 10}
    )
    
    # 收集检索结果作为上下文
    print(f"查询关键词: '{query}'")
    print(f"找到 {len(results[0])} 条相关结果")
    
    context_texts = []
    for i, result in enumerate(results[0]):
        text = result["entity"].get("text", "无文本内容")
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
        
        # 从文件名提取集合名（翻译为英文）
        filename = os.path.basename(filepath)
        raw_name = os.path.splitext(filename)[0]
        translator = ChatOpenAI(
            model="GLM-5V-Turbo",
            api_key=os.environ["ZHIPUAI_API_KEY"],
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
        translate_msg = [
            SystemMessage(content="You are a translator. Translate the given Chinese text to English. Output ONLY the English translation, using underscores instead of spaces, all lowercase. No explanation."),
            HumanMessage(content=raw_name)
        ]
        translated = translator.invoke(translate_msg).content.strip()
        collection_name = re.sub(r'[^a-z0-9_]', '_', translated.lower())
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
        search_project_contracts("JDK怎么安装", collection_name=collection_name, limit=5)
    else:
        print(f"文件不存在: {filepath}")