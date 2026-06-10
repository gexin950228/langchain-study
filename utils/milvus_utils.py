import os
import configparser
from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client.index import IndexParams


def load_milvus_config(config_path: str = None) -> dict:
    """从配置文件加载 Milvus 配置
    
    Args:
        config_path: 配置文件路径，默认为项目根目录的 env.ini
    
    Returns:
        Milvus 配置字典，包含 host, port, user, password
    """
    if config_path is None:
        # 默认从项目根目录查找
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(current_dir), "env.ini")
    
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    config.read(config_path, encoding='utf-8')
    
    milvus_config = {
        "host": config.get("milvus", "host", fallback="localhost"),
        "port": config.getint("milvus", "port", fallback=19530),
        "user": config.get("milvus", "access_key", fallback="").strip('\"'),
        "password": config.get("milvus", "secret_key", fallback="").strip('\"')
    }
    return milvus_config


def get_milvus_client(config_path: str = None) -> MilvusClient:
    """获取 Milvus 客户端连接
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        MilvusClient 实例
    """
    milvus_config = load_milvus_config(config_path)
    uri = f"http://{milvus_config['host']}:{milvus_config['port']}"
    
    client = MilvusClient(
        uri=uri,
        user=milvus_config['user'],
        password=milvus_config['password']
    )
    print(f"成功连接到 Milvus: {uri}")
    return client


def save_to_milvus(texts: list, embeddings: list, collection_name: str = "embedding_demo", 
                   overwrite: bool = True, config_path: str = None) -> None:
    """将文本和向量存入 Milvus
    
    Args:
        texts: 文本列表
        embeddings: 向量列表，与 texts 一一对应
        collection_name: 集合名称
        overwrite: 如果集合已存在是否覆盖
        config_path: 配置文件路径
    """
    if not texts or not embeddings:
        raise ValueError("texts 和 embeddings 不能为空")
    
    if len(texts) != len(embeddings):
        raise ValueError("texts 和 embeddings 长度必须一致")
    
    try:
        client = get_milvus_client(config_path)
        
        # 检查集合是否存在，如果存在则删除
        if client.has_collection(collection_name):
            if overwrite:
                client.drop_collection(collection_name)
                print(f"已删除已存在的集合: {collection_name}")
            else:
                raise ValueError(f"集合 {collection_name} 已存在")
        
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
        
    except Exception as e:
        print(f"操作 Milvus 失败: {e}")
        raise


def search_milvus(query_embedding: list, collection_name: str = "embedding_demo", 
                  limit: int = 5, config_path: str = None) -> list:
    """在 Milvus 中搜索相似向量
    
    Args:
        query_embedding: 查询向量
        collection_name: 集合名称
        limit: 返回结果数量
        config_path: 配置文件路径
    
    Returns:
        搜索结果列表，包含 text 字段和距离
    """
    try:
        client = get_milvus_client(config_path)
        
        # 搜索 Milvus
        results = client.search(
            collection_name=collection_name,
            data=[query_embedding],
            limit=limit,
            output_fields=["text"],
            anns_field="vector",
            search_params={"nprobe": 10}
        )
        
        # 整理结果
        search_results = []
        for result in results[0]:
            search_results.append({
                "text": result["entity"].get("text", ""),
                "distance": result["distance"]
            })
        
        return search_results
    
    except Exception as e:
        print(f"搜索 Milvus 失败: {e}")
        raise