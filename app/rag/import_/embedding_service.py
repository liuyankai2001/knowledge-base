import json
from pathlib import Path
from typing import Any

from app.rag.import_.config import EMBEDDING_BATCH_SIZE
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_providers
from app.process.import_.agent.state import ImportGraphState


def get_data_and_validates(state: ImportGraphState) -> tuple[list[dict[str, Any]], str]:
    md_path = state.get("md_path")
    item_name = state.get("item_name")
    chunks = state.get("chunks")
    if not item_name:
        if md_path and Path(md_path).exists():
            item_name = Path(md_path).stem
        else:
            item_name = "default_item_name"
        logger.warning(f"item_name,给与默认值:{item_name}")

    if not chunks:
        if md_path:
            chunks_json_obj: Path = Path(md_path).with_name(f"{Path(md_path).stem}.json")
            chunks = json.loads(chunks_json_obj.read_text(encoding="utf-8"))
        if not chunks:
            logger.error(f"chunks为空，读取本地备份文件依然为空，业务无法继续进行!")
            raise ValueError(f"chunks为空，读取本地备份文件依然为空，业务无法继续进行!")
    return chunks, item_name


def batch_generate_embeddings(chunks:list[dict[str,Any]], item_name:str, batch_number:int= EMBEDDING_BATCH_SIZE)->list[dict[str,Any]]:
    length = len(chunks)
    for index in range(0,length,batch_number):
        current_chunks = chunks[index:index+batch_number]
        current_content_list = [f"主体:{item_name},内容:{chunk.get('content')}" for chunk in current_chunks]
        result = llm_providers.generate_embeddings(current_content_list)
        for idx,current_chunk in enumerate(current_chunks):
            current_chunk['dense_vector'] = result.get('dense')[idx]
            current_chunk['sparse_vector'] = result.get('sparse')[idx]

    return chunks

def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """
    向量化服务：
    1. 读取 chunks
    2. 生成 dense_vector / sparse_vector
    3. 将向量结果补充回 chunks
    """
    chunks, item_name = get_data_and_validates(state)

    embeddings_content = batch_generate_embeddings(chunks, item_name, EMBEDDING_BATCH_SIZE)

    state['embeddings_content'] = embeddings_content

    return state
