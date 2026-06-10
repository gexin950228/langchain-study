import os
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from langchain_community.document_loaders import (
    Docx2txtLoader, TextLoader, UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader, CSVLoader, PyPDFLoader,
    UnstructuredMarkdownLoader, JSONLoader, BSHTMLLoader, UnstructuredPowerPointLoader,
    UnstructuredRTFLoader, UnstructuredEPubLoader
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import DataType
from pymilvus.milvus_client.index import IndexParams

# 导入 utils 模块中的 Milvus 工具函数
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.milvus_utils import get_milvus_client

os.environ["ZHIPUAI_API_KEY"] = "69695283f7034931b87220e76ef4f6f4.m11eKchDK9Ac7ZIe"
BATCH_SIZE = 64
MILVUS_URI = "http://172.27.176.25:19530"
MILVUS_USER = "root"
MILVUS_PASSWORD = "ufa4A$hiTyTeP@V$a"
class FileLoader:
    FILE_LOADERS = {
        '.doc':  UnstructuredWordDocumentLoader,
        '.docx': Docx2txtLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.xls':  UnstructuredExcelLoader,
        '.csv':  CSVLoader,
        '.pdf':  PyPDFLoader,
        '.txt':  TextLoader,
        '.md':   UnstructuredMarkdownLoader,
        '.json': JSONLoader,
        '.html': BSHTMLLoader,
        '.htm':  BSHTMLLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.rtf':  UnstructuredRTFLoader,
        '.epub': UnstructuredEPubLoader,
    }

    @classmethod
    def get_supported_extensions(cls) -> set:
        return set(cls.FILE_LOADERS.keys())

    @classmethod
    def load(cls, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ('.xls', '.xlsx'):
            try:
                import pandas as pd
                engine = 'xlrd' if ext == '.xls' else 'openpyxl'
                xls = pd.ExcelFile(file_path, engine=engine)
                print(f"    [FileLoader] 发现 {len(xls.sheet_names)} 个sheet: {xls.sheet_names}")
                all_sheets = []
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name, engine=engine)
                    sheet_text = f"\n=== Sheet: {sheet_name} ===\n"
                    sheet_text += df.to_string(index=False)
                    all_sheets.append(sheet_text)
                result = "\n".join(all_sheets)
                print(f"    [FileLoader] pandas+{engine}加载{ext}文件成功，共{len(xls.sheet_names)}个sheet，内容长度: {len(result)}")
                return result
            except ImportError:
                print(f"    [FileLoader] {engine}未安装，回退到LangChain Loader")
            except Exception as e:
                print(f"    [FileLoader] pandas加载失败({e})，回退到LangChain Loader")
        loader_cls = cls.FILE_LOADERS.get(ext)
        if not loader_cls:
            raise ValueError(f"不支持的文件格式: {ext}，支持格式: {cls.get_supported_extensions()}")
        kwargs = {}
        if ext == '.txt':
            kwargs['encoding'] = 'utf-8'
        elif ext == '.json':
            kwargs['jq_schema'] = '.'
        try:
            loader = loader_cls(file_path, **kwargs)
            docs = loader.load()
            result = "\n".join(doc.page_content for doc in docs)
            print(f"    [FileLoader] {loader_cls.__name__}加载成功，共{len(docs)}块，内容长度: {len(result)}")
            return result
        except Exception as e:
            print(f"    [FileLoader] {loader_cls.__name__}加载失败: {e}")
            raise

ALLOWED_EXTENSIONS = FileLoader.get_supported_extensions()

_translator = None

def translate_to_english(chinese_text: str) -> str:
    global _translator
    if _translator is None:
        _translator = ChatOpenAI(
            model="GLM-5V-Turbo",
            api_key=os.environ.get("ZHIPUAI_API_KEY", ""),
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
    messages = [
        SystemMessage(content="Translate to English. Output ONLY lowercase English with underscores replacing spaces. No explanation."),
        HumanMessage(content=chinese_text)
    ]
    return re.sub(r'[^a-z0-9_]', '_', _translator.invoke(messages).content.strip().lower())

def find_all_files(directory: str) -> list:
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS:
                file_paths.append(os.path.join(root, filename))
    return file_paths

def get_collection_name(file_path: str) -> str:
    dir_name = os.path.basename(os.path.dirname(file_path))
    return translate_to_english(dir_name)

def get_partition_name(file_path: str) -> str:
    name_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    return translate_to_english(name_without_ext)



MAX_TEXT_LENGTH = 65535
MAX_CHUNK_SIZE = 4000

def truncate_text_by_bytes(text: str, max_bytes: int = 65535) -> str:
    """按字节长度截断文本（Milvus VARCHAR按字节存储）"""
    text_bytes = text.encode('utf-8')
    if len(text_bytes) <= max_bytes:
        return text
    
    # 找到合适的截断位置，避免截断在多字节字符中间
    truncated_bytes = text_bytes[:max_bytes]
    # 尝试找到完整字符的边界
    for i in range(min(4, max_bytes), 0, -1):
        try:
            truncated = truncated_bytes[:max_bytes - i + 1].decode('utf-8')
            return truncated
        except UnicodeDecodeError:
            continue
    
    # 如果都失败，直接截断
    return text_bytes[:max_bytes - 1].decode('utf-8', errors='ignore')

def semantic_chunk(content: str) -> list:
    embedding = ZhipuAIEmbeddings(model="text_embedding")
    text_splitter = SemanticChunker(embeddings=embedding)
    split_docs = text_splitter.create_documents([content])
    paragraphs = [doc.page_content for doc in split_docs]
    if len(paragraphs) <= 1 and len(content) > MAX_CHUNK_SIZE * 2:
        print(f"    [语义切割] 仅产出{len(paragraphs)}块，回退到字符切割")
        fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=200)
        paragraphs = fallback_splitter.split_text(content)
        print(f"    [字符切割] 切割为 {len(paragraphs)} 个块")
    result = []
    for p in paragraphs:
        byte_length = len(p.encode('utf-8'))
        if byte_length > MAX_TEXT_LENGTH:
            print(f"    [截断] 单块字节长度{byte_length}超过{MAX_TEXT_LENGTH}，进行截断")
            p = truncate_text_by_bytes(p, MAX_TEXT_LENGTH)
        if len(p) > MAX_TEXT_LENGTH:
            print(f"    [截断] 单块长度{len(p)}超过{MAX_TEXT_LENGTH}，进行截断")
            p = p[:MAX_TEXT_LENGTH]
        result.append(p)
    return result

def get_batch_embeddings(texts: list) -> list:
    embedding = ZhipuAIEmbeddings(model="text_embedding")
    all_vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        print(f"  向量化第 {i // BATCH_SIZE + 1} 批，共 {len(batch)} 条")
        all_vectors.extend(embedding.embed_documents(batch))
    return all_vectors

def batch_embedding_save(file_paths: list):
    client = get_milvus_client()
    print("连接Milvus成功")
    collection_name = get_collection_name(file_paths[0])
    print(f"目标集合: {collection_name}")
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print("已删除已存在的集合")
    first_file_done = False
    global_id = 0
    results_summary = []
    for idx, file_path in enumerate(file_paths):
        filename = os.path.basename(file_path)
        print(f"\n{'='*60}")
        print(f"[{idx + 1}/{len(file_paths)}] 处理文件: {filename}")

        file_result = {"file": filename, "status": "success", "details": ""}
        try:
            content = FileLoader.load(file_path)
            if len(content.strip()) == 0:
                raise ValueError("文件内容为空!")
            paragraphs = semantic_chunk(content)
            if len(paragraphs) == 0:
                raise ValueError("语义切割结果为空!")
            print(f"  ✅ 语义切割为 {len(paragraphs)} 个块")

            embeddings = get_batch_embeddings(paragraphs)
            print(f"  ✅ 向量化完成，维度: {len(embeddings[0])}")

            if not first_file_done:
                schema = client.create_schema(auto_id=False)
                schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
                schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=len(embeddings[0]))
                client.create_collection(collection_name, schema=schema)
                index_params = IndexParams()
                index_params.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="L2", params={"nlist": 128})
                client.create_index(collection_name=collection_name, index_params=index_params)
                first_file_done = True
                print(f"  ✅ 集合创建完成")

            partition_name = get_partition_name(file_path)
            if not client.has_partition(collection_name, partition_name):
                client.create_partition(collection_name, partition_name)
                print(f"  ✅ 创建 partition: {partition_name}")
            # 最终检查：确保文本字节长度不超过 VARCHAR 最大长度
            MAX_VARCHAR_LENGTH = 65535
            data = []
            for i in range(len(paragraphs)):
                text = paragraphs[i]
                byte_length = len(text.encode('utf-8'))
                if byte_length > MAX_VARCHAR_LENGTH:
                    print(f"    [最终截断] 文本块{i+1}字节长度{byte_length}超过{MAX_VARCHAR_LENGTH}，进行截断")
                    text = truncate_text_by_bytes(text, MAX_VARCHAR_LENGTH)
                data.append({"id": global_id + i, "text": text, "vector": embeddings[i]})
            
            global_id += len(paragraphs)
            result = client.insert(collection_name=collection_name, data=data, partition_name=partition_name)
            print(f"  ✅ 已插入 {len(data)} 条数据到 partition[{partition_name}]")
            file_result["details"] = f"{len(data)}条数据 → partition[{partition_name}]"

        except Exception as e:
            print(f"  ❌ 处理文件失败: {e}")
            file_result["status"] = "failed"
            file_result["details"] = str(e)
        finally:
            results_summary.append(file_result)

    client.load_collection(collection_name=collection_name)
    stats = client.get_collection_stats(collection_name)
    print(f"\n{'='*60}")
    print(f"【处理结果汇总】")
    print(f"{'='*60}")
    success_count = sum(1 for r in results_summary if r["status"] == "success")
    fail_count = sum(1 for r in results_summary if r["status"] == "failed")
    print(f"成功: {success_count}/{len(results_summary)}, 失败: {fail_count}/{len(results_summary)}")
    print(f"集合统计信息: {stats}\n")
    for r in results_summary:
        icon = "✅" if r["status"] == "success" else "❌"
        print(f"  {icon} {r['file']}: {r['details']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="批量向量化文件并存入Milvus")
    parser.add_argument("-d", "--directory", help="要处理的目录路径", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        print(f"目录不存在: {args.directory}")
        exit(1)

    file_paths = find_all_files(args.directory)
    print(f"共找到 {len(file_paths)} 个文件:\n")
    for f in file_paths:
        print(f"  {f}")
    batch_embedding_save(file_paths)