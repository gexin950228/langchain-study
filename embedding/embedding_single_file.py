import os
import warnings

# 忽略 langchain-community 过时警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client.index import IndexParams

# 设置智谱API密钥
os.environ["ZHIPUAI_API_KEY"] = "69695283f7034931b87220e76ef4f6f4.m11eKchDK9Ac7ZIe"

# 智谱API限制：每次最多处理64条
BATCH_SIZE = 64

def get_batch_embeddings(texts: list) -> list:
    """批量获取文本向量（分批处理）"""
    # 使用正确的模型名称
    embedding = ZhipuAIEmbeddings(model="text_embedding")
    all_vectors = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        print(f"处理第 {i//BATCH_SIZE + 1} 批，共 {len(batch)} 条")
        vectors = embedding.embed_documents(batch)
        all_vectors.extend(vectors)
    
    return all_vectors

# 修改 save_to_milvus 函数，创建包含 text 字段的集合
def save_to_milvus(texts: list, embeddings: list, collection_name: str = "embedding_demo"):
    """将文本和向量存入 Milvus"""
    try:
        # 连接 Milvus（使用 HTTP 协议，端口 19530）
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
    
    # 连接 Milvus
    client = MilvusClient(
        uri="http://172.18.83.231:19530",
        user="root",
        password="ufa4A$hiTyTeP@V$a"
    )
    
    # 搜索 Milvus
    results = client.search(
        collection_name="embedding_demo",
        data=[query_embedding],
        limit=limit,
        output_fields=["text"]
    )
    
    # 输出结果
    print(f"查询关键词: '{query}'")
    print(f"找到 {len(results[0])} 条相关结果:\n")
    
    for i, result in enumerate(results[0]):
        text = result["entity"].get("text", "无文本内容")
        distance = result["distance"]
        print(f"{i+1}. 相似度: {1 - distance:.4f}")  # 转换为相似度
        print(f"   内容: {text}")
        print()

if __name__ == "__main__":
    pwd = os.getcwd()
    filepath_path = os.path.join(pwd, "files\\embedding-demo.txt")
    
    if os.path.exists(filepath_path):
        print(f"文件路径: {filepath_path}")
        
        # 读取文件内容
        with open(filepath_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        print(f"文件内容长度: {len(content)} 字符")
        
        # 简单分块处理（按段落分割）
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        print(f"分割为 {len(paragraphs)} 个段落")
        
        # 调用智谱embedding模型进行向量化
        print("\n正在进行向量化...")
        embeddings = get_batch_embeddings(paragraphs)
        
        print(f"\n向量化完成!")
        print(f"生成了 {len(embeddings)} 个向量")
        print(f"每个向量维度: {len(embeddings[0])}")
        
        # 打印第一个向量的前20个元素
        print(f"\n第一个段落的向量（前20维）:")
        print(embeddings[0][:20])
        
        # 将向量存入 Milvus
        print("\n正在将向量存入 Milvus...")
        save_to_milvus(paragraphs, embeddings)
        print("\n数据已成功存入 Milvus!")
        
        # 调用检索函数搜索项目组联系人信息
        search_project_contacts("应用服务启停", 5)
    else:
        print(f"文件不存在: {filepath_path}")