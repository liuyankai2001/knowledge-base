import json
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

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
    return state
