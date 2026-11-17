import copy
from typing import TypedDict


class ImportGraphState(TypedDict):
    # 第一个节点
    task_id:str
    local_file_path:str

    # 文件类型标识
    md_path:str
    is_md_read_enabled:bool

    pdf_path:str
    is_pdf_read_enabled:bool

    file_title:str # 文件名

    # 输出文件地址
    local_dir:str # 存储生成文件地址 文件夹

    md_content:str # md文件内容

    chunks:list # 装切割后的文本块 dict

    item_name:str # 主题 file_title做兜底

    embeddings_content:list # 带有向量的切块

default_state: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": [],
}

def create_default_state(**kwargs) -> ImportGraphState:
    new_state = copy.deepcopy(default_state)
    new_state.update(kwargs)
    return new_state

def get_default_state()->ImportGraphState:
    return copy.deepcopy(default_state)