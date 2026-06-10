import os
<<<<<<< HEAD
import sys
import configparser
import warnings

# 添加项目根目录到路径，以便导入 utils 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 忽略 langchain-community 过时警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

=======
import warnings
# 忽略 langchain-community 过时警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
>>>>>>> f9ba5d08b4c35630a8f536eeb8f2032fdf1229fe
from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client.index import IndexParams
<<<<<<< HEAD

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

=======
>>>>>>> f9ba5d08b4c35630a8f536eeb8f2032fdf1229fe
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

<<<<<<< HEAD
# 使用 utils/milvus_utils 中的 save_to_milvus 函数（已通过 import 导入）


# 创建向量检索函数（使用 utils/milvus_utils 中的 search_milvus 函数）
=======
# 修改 save_to_milvus 函数，创建包含 text 字段的集合
def save_to_milvus(texts: list, embeddings: list, collection_name: str = "embedding_demo"):
    """将文本和向量存入 Milvus"""
    try:
        # 连接 Milvus（使用 HTTP 协议，端口 19530）
        client = MilvusClient(
            uri="http://172.27.176.25:19530",
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
        
        # 准备数据（包含 text 字段）
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
        print(f"已插入 {len(data)} 条数据（包含文本内容），插入结果: {result}")
        
        # 为向量字段创建索引（Milvus 需要先创建索引才能加载和搜索）
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
        # 加载集合到内存（Milvus 需要先加载才能搜索）
        client.load_collection(collection_name=collection_name)
        print(f"集合已加载到内存")
        
        # 查看集合统计信息
        stats = client.get_collection_stats(collection_name)
        print(f"集合统计信息: {stats}")
        
    except Exception as e:
        print(f"连接 Milvus 失败: {e}")
        print("将使用本地检索模式")


# 创建向量检索函数
>>>>>>> f9ba5d08b4c35630a8f536eeb8f2032fdf1229fe
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
    
<<<<<<< HEAD
    # 使用 utils/milvus_utils 中的 search_milvus 函数搜索
    results = search_milvus(query_embedding, collection_name="embedding_demo", limit=limit)
    
    # 收集检索结果作为上下文
    print(f"查询关键词: '{query}'")
    print(f"找到 {len(results)} 条相关结果:\n")
    
    context_texts = []
    for i, result in enumerate(results):
        text = result["text"]
=======
    # 连接 Milvus
    client = MilvusClient(
        uri="http://172.27.176.25:19530",
        user="root",
        password="ufa4A$hiTyTeP@V$a"
    )
    
    # 搜索 Milvus
    results = client.search(
        collection_name="embedding_demo",
        data=[query_embedding],
        limit=limit,
        output_fields=["text"],
        anns_field="vector",
        search_params={"nprobe": 10}
    )
    
    # 收集检索结果作为上下文
    print(f"查询关键词: '{query}'")
    print(f"找到 {len(results[0])} 条相关结果:\n")
    
    context_texts = []
    for i, result in enumerate(results[0]):
        text = result["entity"].get("text", "无文本内容")
>>>>>>> f9ba5d08b4c35630a8f536eeb8f2032fdf1229fe
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
        save_to_milvus(paragraphs, embeddings)
        print("\n数据已成功存入 Milvus!")
        
        # 调用检索函数搜索项目组联系人信息
        search_project_contacts("redis服务的地址和启动方式是什么", 5)
    else:
        print(f"文件不存在: {filepath_path}")