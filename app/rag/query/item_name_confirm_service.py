from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.vector_store.milvus_gateway import milvus_gateway
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


def search_by_item_name(item_names:list[str]):
    """
    向量数据库搜索
    :param item_names:
    :return:
    """
    final_result = {}
    # 1.循环LLM查询到item_names的列表->item_name
    item_names_vector = llm_providers.generate_embeddings(item_names)
    for index in range(len(item_names)):
        # 2.item_name转成对应稀疏和稠密向量
        item_name = item_names[index]
        item_name_dense = item_names_vector["dense"][index]
        item_name_sparse = item_names_vector["sparse"][index]
        # 3.稀疏和稠密向量进行混合检索
        reqs = milvus_gateway.create_requests(item_name_dense,item_name_sparse,limit=5*2)
        results = milvus_gateway.hybrid_search(   # [[],[]]
            collection_name=milvus_gateway.item_name_collection_name,
            reqs=reqs,
            ranker_weights=(0.5,0.5),
            norm_score=True,
            output_fields=['item_name']
        )
        # 4.处理混合检索的结果
        item_name_milvus_list = []
        if len(results[0]) > 0:
            for item in results[0]:
                item_name_milvus_list.append({
                    "item_name":item.get('entity').get('item_name'),
                    "score":item.get('distance')
                })

        # 5.循环完以后得结果最终返回
        final_result[item_name] = item_name_milvus_list
    return final_result


def select_item_names(milvus_result:dict[str,list[dict]]):
    """ 根据分数，明确确定和可选的item_name """
    confirmed_list = []
    option_list = []
    for item_name,milvus_result_list_dict in milvus_result.items():
        high_score_list = [item for item in milvus_result_list_dict if item.get("score",0) >= 0.7]
        md_score_list = [item for item in milvus_result_list_dict if 0.6 <= item.get("score",0) < 0.7]
        if len(high_score_list) > 0:
            confirmed_list.append(high_score_list[0])
            logger.info(f"模型识别item_name:{item_name}，有对应向量数据库中确认的item_name:{high_score_list[0].get('item_name')}")
            continue
        if len(md_score_list) > 0:
            option_list.extend(md_score_list[:2])
            logger.info(
                f"模型识别item_name:{item_name},没有对应向量数据库中确认的item_name,"
                f"但是有可选的:{','.join([item.get('item_name') for item in md_score_list[:2]])}"
            )
            continue
    return {
        "confirmed_list":confirmed_list,
        "option_list":option_list
    }


def apply_item_name_result(state:QueryGraphState, list_dict:dict[str,list], rewritten_query:str):
    confirmed_list = list_dict.get("confirmed_list",[])
    option_list = list_dict.get("option_list",[])

    if confirmed_list:
        state['item_names'] = [item.get('item_name') for item in confirmed_list]
        state['rewritten_query'] = rewritten_query
        if "answer" in state:
            state['answer'] = None
        return
    if option_list:
        state['answer'] = f"本次提问没有确认的主体，但是有相似可选的：{','.join([item.get("item_name") for item in option_list])},请您再次确认！"
        return
    state["answer"] = "本次问题没有关联到任何主体，有没有相似可选的主体！请您明确主体再提问!"
    return


def save_history_message(state:QueryGraphState):
    history_repository.save_message(
        session_id=state.get("session_id"),
        role="user",
        text=state.get("original_query"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names",[]),
        image_urls=[]
    )



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

    list_dict = {}
    if len(result.get("item_names",[]))>0:
        # 4.进行向量数据库的搜索 llm - 分析 -> item_names -> milvus中进行查询 -> 打分
        milvus_result:dict[str,list[dict]] = search_by_item_name(result.get("item_names",[]))
        # 5.根据打分确定两个列表 确认列表 可选列表
        list_dict:dict[str,list] = select_item_names(milvus_result)
    # 6.确定和可选列表修改state answer item_names rewritten_query
    apply_item_name_result(state,list_dict,result.get("rewritten_query"))
    # 7.记录提问的聊天记录
    save_history_message(state)
    return state