from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger
from app.infra.persistence.history_repository import history_repository
from app.infra.llm.providers import llm_providers

def get_data_and_validates(state:QueryGraphState):
    session_id = state.get("session_id")
    original_query = state.get("original_query")
    if not session_id or not original_query:
        logger.error(f"session_id或者original_query未空,业务无法继续进行,提前终止!")
        raise ValueError(f"session_id或者original_query未空,业务无法继续进行,提前终止!")

    return session_id,original_query


def get_history_messages_and_context(session_id:str):
    # 1.获取最近10条历史记录
    message_list:list[dict] = history_repository.list_recent(session_id=session_id,limit=10)
    if not message_list or len(message_list) == 0:
        logger.warning(f"当前会话:{session_id}没有历史对话记录!提前跳出,history_text未空!")
        return "无对话记录"

    final_message_list = [
        item
        for item in message_list if len(item.get("item_names",[])) > 0
    ]
    if not final_message_list or len(final_message_list) == 0:
        logger.warning(f"当前会话:{session_id}没有有效的历史对话记录!提前跳出,history_text未空!")
        return "无有效对话记录"

    # 拼接histor_text
    history_text = ""
    for index,item in enumerate(final_message_list,start=1):
        history_text += (
            f"序号:{index},{'提问:' if item.get('role') == 'user' else '回答:'}"
            f"{item.get('rewritten_query') if item.get('role') == 'user' else item.get('text')[:50]},"
            f"关联主体: {','.join(item.get('item_names'))}\n"
        )
    return history_text


def call_llm_item_name_and_rewritten(history_text:str, original_query:str):

    # 1.加载模型对象
    json_llm_client = llm_providers.chat(json_mode=True)
    # 2.加载提示词
    history_prompt_text = load_prompt("rewritten_query_and_itemnames",query=original_query,history_text=history_text)
    # 3.包装提示词对象
    history_prompt_messages = [
        HumanMessage(content=history_prompt_text)
    ]
    # 4.创建调用链
    chains = json_llm_client | JsonOutputParser()
    # 5.调用获取结果
    result_dict = chains.invoke(history_prompt_messages)
    # 6.参数校验赋予默认值
    if "item_names" not in result_dict:
        result_dict["item_names"] = []
    if "rewritten_query" not in result_dict:
        result_dict["rewritten_query"] = original_query
    # 7.返回结果
    return result_dict


def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    """
    意图确认服务：
    1. 结合历史对话提取商品名
    2. 将模糊问题改写为完整独立的精准问题
    3. 在 Milvus 向量库中进行混合搜索
    4. 根据评分高低自动对齐标准型号，或生成反问让用户手动确认
    5. 同步历史记录到 MongoDB
    """
    # 1.校验和获取参数
    session_id, original_query = get_data_and_validates(state)
    # 2.获取历史信息和拼接上下文
    history_text = get_history_messages_and_context(session_id)
    # 3.调用模型识别item_names和重写的问题
    result:dict = call_llm_item_name_and_rewritten(history_text,original_query)

    return state