import json
import requests

json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()

from langchain_text_splitters import RecursiveJsonSplitter

splitter = RecursiveJsonSplitter(max_chunk_size=300)
json_chunks = splitter.split_json(json_data)  # 返回字典列表，而非 Document 对象

json_chunks_document = splitter.create_documents(json_chunks)

print(f"共切割成 {len(json_chunks)} 个 JSON 块")
print("=" * 50)
# for i, chunk in enumerate(json_chunks):
#     print(f"\n--- JSON块 {i+1} ---")
#     # 将字典转换为格式化的 JSON 字符串
#     json_str = json.dumps(chunk, indent=2, ensure_ascii=False)
#     print(json_str[:300] + "..." if len(json_str) > 300 else json_str)

for chunk in json_chunks_document[:10]:
    print(chunk.page_content)