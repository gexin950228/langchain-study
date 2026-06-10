import os
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 导入 utils 模块中的 Milvus 工具函数
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.milvus_utils import get_milvus_client

os.environ["ZHIPUAI_API_KEY"] = "69695283f7034931b87220e76ef4f6f4.m11eKchDK9Ac7ZIe"
COLLECTION_NAME = "group_power_industry_operation_monitoring_system"

def get_query_embedding(query: str) -> list:
    embedding = ZhipuAIEmbeddings(model="text_embedding")
    return embedding.embed_query(query)

def ask_llm(question: str, context: str) -> str:
    llm = ChatOpenAI(
        model="GLM-5V-Turbo",
        api_key=os.environ.get("ZHIPUAI_API_KEY", ""),
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    messages = [
        SystemMessage(content="你是一个专业的技术文档助手。请根据以下检索到的文档内容回答用户问题。如果文档中没有相关信息，请如实说明。"),
        HumanMessage(content=f"参考文档内容：\n{context}\n\n用户问题：{question}")
    ]
    return llm.invoke(messages).content.strip()

def search_all_partitions(collection_name: str, query: str, limit: int = 5):
    client = get_milvus_client()
    print(f"连接Milvus成功")

    if not client.has_collection(collection_name):
        print(f"集合不存在: {collection_name}")
        return
    if not client.get_load_state(collection_name)["state"] == "Loaded":
        client.load_collection(collection_name)
        print(f"集合已加载到内存")

    partitions = client.list_partitions(collection_name)
    print(f"集合 {collection_name} 共有 {len(partitions)} 个partition:\n  {partitions}\n")

    query_embedding = get_query_embedding(query)
    print(f"查询向量化完成，维度: {len(query_embedding)}\n")

    all_results = []
    for partition_name in partitions:
        try:
            results = client.search(
                collection_name=collection_name,
                data=[query_embedding],
                limit=limit,
                output_fields=["text"],
                anns_field="vector",
                partition_names=[partition_name],
            )
            if results and results[0]:
                for hit in results[0]:
                    all_results.append({
                        "text": hit.get("entity", {}).get("text", ""),
                        "distance": hit.get("distance", 0),
                        "partition": partition_name,
                    })
        except Exception as e:
            print(f"  检索 partition[{partition_name}] 失败: {e}")

    all_results.sort(key=lambda x: x["distance"])
    print(f"共检索到 {len(all_results)} 条结果\n")
    context_texts = [r["text"] for r in all_results[:limit]]
    context = "\n---\n".join(context_texts) if context_texts else "未检索到相关内容"

    answer = ask_llm(query, context)
    print(f"\n{'='*60}")
    print(f"问题: {query}")
    print(f"\n答案:\n{answer}")
    return answer

if __name__ == "__main__":
    query = input("请输入您的问题 (默认: 查找power_operation数据库的数据库服务器是哪些): ").strip()
    if not query:
        query = "查找power_operation数据库的数据库服务器是哪些"
    search_all_partitions(COLLECTION_NAME, query, limit=10)