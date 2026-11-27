from datetime import datetime
from mimetypes import guess_type
from pathlib import Path
import sys
import uuid

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schema.query_schema import HealthResponseSchema, QueryRequestSchema, QueryNotStreamResponseSchema, \
    QueryStreamResponseSchema, HistoryClearResponseSchema, HistoryItemResponseSchema, HistoryListResponseSchema
from app.infra.persistence.history_repository import history_repository
# from app.infra.persistence.history_repository import history_repository
#
# from app.api.schema.query_schema import HealthResponseSchema, QueryRequestSchema, QueryStreamResponseSchema, \
#     QueryNotStreamResponseSchema, HistoryClearResponseSchema, HistoryListResponseSchema, HistoryItemResponseSchema
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.infra.config.providers import settings
from app.process.query.agent.main_graph import query_app
from app.process.query.agent.state import create_query_default_state, QueryGraphState
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    clear_task,
    get_done_task_list,
    get_task_result,
    update_task_status,
)

# 定义fastapi对象
app = FastAPI(
    title=settings.query_app_name,
    description="描述,进行rag查询的服务对象",
    version="0.2.0"
)

# 跨域处理 [前端和后端可能不在一个服务器,非同源,浏览器认为跨域,拦截了 ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)


@app.get("/html")
def query_html():
    query_html_obj = PROJECT_ROOT / "app" / "resources" / "html" / "chat.html"
    return FileResponse(
        path=str(query_html_obj),
        media_type=guess_type(query_html_obj.name)[0]
    )


def invoke_query_graph(session_id: str, original_query: str, is_stream: bool):
    try:
        clear_task(session_id)
        if is_stream:
            create_sse_queue(session_id)

        update_task_status(session_id, TASK_STATUS_PROCESSING, push_queue=is_stream)
        query_state = create_query_default_state(session_id=session_id, original_query=original_query,
                                                 is_stream=is_stream)
        result_state = query_app.invoke(query_state)
        update_task_status(session_id, TASK_STATUS_COMPLETED, push_queue=is_stream)
        if is_stream:
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": result_state.get("answer"),
                    "status": "completed",
                    "image_urls": result_state.get("image_urls", [])
                }
            )
        return result_state

    except Exception as e:
        logger.exception(f"执行查询流程报错，错误信息:{str(e)}")
        update_task_status(session_id, TASK_STATUS_FAILED)


@app.get("/health")
def health():
    logger.info(f"完成健康检查!{datetime.now()}")
    return HealthResponseSchema(
        code=200,
        messages="完成健康检查!"
    )


@app.post("/query")
def query(backgroundtasks: BackgroundTasks, query_params: QueryRequestSchema):
    # 1.获取参数
    query = query_params.query
    session_id = query_params.session_id or uuid.uuid4()
    # 2.封装执行图对象
    is_stream = query_params.is_stream

    # 3.判断是不是流式
    if is_stream:  # 4.是 异步执行图方法
        backgroundtasks.add_task(
            invoke_query_graph,
            session_id=session_id,
            original_query=query,
            is_stream=is_stream
        )
        return QueryStreamResponseSchema(
            message=f"已经开始:{query}任务查询!",
            session_id=session_id
        )

    else:  # 5.不是 直接调用图方法
        state: QueryGraphState = invoke_query_graph(
            session_id=session_id,
            original_query=query,
            is_stream=is_stream
        )
        done_list = get_done_task_list(session_id)
        return QueryNotStreamResponseSchema(
            message=f"完成{query}所有内容检索!",
            session_id=session_id,
            answer=state.get("answer"),
            done_list=done_list,
            image_urls=state.get("image_urls", [])
        )


@app.get("/stream/{session_id}")
def stream(session_id: str, request: Request):
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream"
    )

@app.delete("/history/{session_id}")
def clear_history(session_id:str):
    deleted_count = history_repository.clear_session(session_id=session_id)
    return HistoryClearResponseSchema(
        message=f"session_id:{session_id}历史聊天记录已经清空!",
        deleted_count=deleted_count
    )

@app.get("/history/{session_id}")
def get_history(session_id:str,limit:int=10):
    history_list:list[dict] = history_repository.list_recent(session_id=session_id,limit=limit)
    return HistoryListResponseSchema(
        session_id=session_id,
        items=[
            HistoryItemResponseSchema(
                id=str(item.get('_id')),
                session_id=session_id,
                role=item.get("role"),
                text=item.get("text"),
                rewritten_query=item.get("rewritten_query"),
                item_names=item.get("item_names",[]),
                image_urls=item.get("image_urls",[]),
                ts=item.get("ts")
            )
            for item in history_list
        ]
    )



if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)
