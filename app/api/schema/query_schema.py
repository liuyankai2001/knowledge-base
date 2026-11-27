from typing import Any

from pydantic import BaseModel,Field

# 健康检查响应json
class HealthResponseSchema(BaseModel):
    code:int = 200
    messages:str = None

# 查询接口的请求json
class QueryRequestSchema(BaseModel):
    query:str
    session_id:str = None
    is_stream:bool = False

# 查询模块流式响应结果
class QueryStreamResponseSchema(BaseModel):
    message:str
    session_id:str

# 查询模块非流式响应结果
class QueryNotStreamResponseSchema(BaseModel):
    message:str
    session_id:str
    answer:str
    done_list:list[str]
    image_urls:list[str]

# 会话记录删除响应
class HistoryClearResponseSchema(BaseModel):
    message:str
    deleted_count:int

class HistoryItemResponseSchema(BaseModel):
    id:str
    session_id:str
    role:str
    text:str
    rewritten_query:str = None
    item_names:list[str] = Field(default_factory=lambda : [])
    image_urls:list[str] = []
    ts:Any

class HistoryListResponseSchema(BaseModel):
    session_id:str
    items:list[HistoryItemResponseSchema] = Field(default_factory=lambda :[])