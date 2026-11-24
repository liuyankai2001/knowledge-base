import json
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNKS_SPLIT_TOP_NUMBER
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_providers


def get_data_and_validates(state: ImportGraphState) -> tuple[list[dict[str, Any]], str]:
    md_path = state.get("md_path")
    file_title = state.get("file_title")
    chunks = state.get("chunks")
    if not file_title:
        if md_path and Path(md_path).exists():
            file_title = Path(md_path).stem
        else:
            file_title = "default_title"
        logger.warning(f"file_title没有值,给与默认值:{file_title}")

    if not chunks:
        if md_path:
            chunks_json_obj: Path = Path(md_path).with_name(f"{Path(md_path).stem}.json")
            chunks = json.loads(chunks_json_obj.read_text(encoding="utf-8"))
        if not chunks:
            logger.error(f"chunks为空，读取本地备份文件依然为空，业务无法继续进行!")
            raise ValueError(f"chunks为空，读取本地备份文件依然为空，业务无法继续进行!")
    return chunks, file_title


def recognize_item_name_by_chunks(chunks, file_title) -> str:
    chat_client = llm_providers.chat()
    system_prompt_text: str = load_prompt("product_recognition_system")
    user_context: str = ""
    used_chunks = chunks[:CHUNKS_SPLIT_TOP_NUMBER]
    """
    context =
        第1 2 3切块:标题:title,内容:content \n
        chunk \n
        chunk \n
        chunk \n
    """
    for index, chunk in enumerate(used_chunks, start=1):
        user_context += f"第{index}部分: 标题为{chunk.get('title')}，内容为:{chunk.get('content')}\n"

    user_prompt_text: str = load_prompt("item_name_recognition", file_title=file_title, context=user_context)
    messages = [
        SystemMessage(content=system_prompt_text),
        HumanMessage(content=user_prompt_text)
    ]

    chains = chat_client | StrOutputParser()

    item_name = chains.invoke(messages)

    if not item_name:
        item_name = file_title
        logger.warning(f"模型未识别到item_name使用file_title:{file_title}赋予默认值!")
    logger.info(f"模型识别到item_name:{item_name}")
    return item_name


def chunk_update_item_name(chunks, item_name):
    for chunk in chunks:
        chunk["item_name"] = item_name
    logger.info(f"完成chunk的item_name属性更新：{item_name}")


def prepared_milvus_item_name_collection():
    item_name_collection_name = milvus_gateway.item_name_collection_name
    milvus_client = milvus_gateway.milvus_client
    # 判断是否有集合
    has_item_name_collection = milvus_client.has_collection(collection_name=item_name_collection_name)

    if has_item_name_collection:
        # 有 True
        logger.info(f"{item_name_collection_name}集合已经存在无需创建!")
        return
        # 3.没有创建集合
    logger.info(f"{item_name_collection_name}集合不存在,开始创建!")
    # 创建: schema field  主键  索引..
    # 准备schema -> 这里可以添加field列信息
    schema = milvus_client.create_schema(
        auto_id=True,  # 主键自增长
        enable_dynamic_field=True,  # schema设置列的信息! True插入了没有提前设定好的列,也可以进行存储
    )
    # 列 名称 类型 长度限制..
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    # FLOAT_VECTOR -> 32 没有启动fp16加速 cpu或者gpu没有加速   FLOAT16_VECTOR -> gpu fp16
    # dim = 维度  -> 根据嵌入式模型生成的维度  bge-m3 1024
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 索引
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE", # 稠密向量相似度：COSINE = IP 归一化l2
        params={
            "M":64, # 每个点最大连接数量
            "efConstruction": 100 # 考虑连接的范围，在这个范围内确定M
        }
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    milvus_client.create_collection(
        collection_name=item_name_collection_name,
        schema=schema,
        index_params=index_params
    )

def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    """
    主体识别服务：
    1. 基于 chunks 构造上下文
    2. 调用 LLM 识别 item_name
    3. 将 item_name 回填到 state 和 chunks
    4. 同步写入主体名称索引
    """
    # 1. 获取并且校验参数（state） -> chunks,file_title
    chunks, file_title = get_data_and_validates(state)
    # 2. 识别item_name
    item_name = recognize_item_name_by_chunks(chunks, file_title)
    chunk_update_item_name(chunks,item_name)

    prepared_milvus_item_name_collection()

    return state
