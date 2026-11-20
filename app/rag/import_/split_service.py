import re
from pathlib import Path
from typing import Any

from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger

def load_markdown_content(state:ImportGraphState)->tuple[str,str]:
    """
    获取参数和校验
    :param state:
    :return:
    """
    md_content = state.get("md_content")
    file_title = state.get("file_title")
    md_path = state.get("md_path")
    if not md_content:
        if md_path and Path(md_path).exists():
            logger.warning(f"md_content内容为空,从备份地址:{md_path}再次读取数据!!")
            md_content = Path(md_path).read_text(encoding='utf-8')
        if not md_content:
            logger.error(f"md_content为空，业务无法继续进行，提前终止!")
            raise ValueError(f"md_content为空，业务无法继续进行，提前终止!")

    if not file_title:
        if md_path and Path(md_path).exists():
            file_title = Path(md_path).stem
        if not file_title:
            file_title = "default"
        state["file_title"] = file_title
        logger.warning(f"file_title为空,启动默认值机制，赋值后:{file_title}")

    md_content = md_content.replace("\r\n","\n").replace("\r","\n")
    state["md_content"] = md_content
    return md_content,file_title


def split_chunks_document(md_content, file_title)->list[dict[str,Any]]:
    """
    根据标题完成语义切割
    :param md_content:
    :param file_title:
    :return:
    """
    md_content_lines:list[str] = md_content.split('\n')

    chunks:list[dict[str,Any]] = [] # [{title,content,file_title}]
    current_title: str|None = None
    current_title_lines:list[str] = []
    is_code_block = False
    title_reg = re.compile(r"^\s*#{1,6}\s.+")

    for line in md_content_lines:
        line_strip = line.strip()
        if not line_strip: # 空行判断
            logger.warning("处理碰到空行!跳过本次处理!")
            continue
        if line_strip.startswith("```") or line_strip.startswith("~~~"):
            is_code_block = not is_code_block
            current_title_lines.append(line_strip)
            continue
        if not is_code_block and title_reg.match(line_strip): # 有效标题
            if current_title_lines and len(current_title_lines)>1:
                chunks.append({
                    "content":"\n".join(current_title_lines),
                    "title":current_title,
                    "file_title":file_title
                })
            if not current_title_lines and len(current_title_lines) > 0:
                current_title_lines.append(line_strip)
            else:
                current_title_lines = [line_strip]
            current_title = line_strip
        else:
            current_title_lines.append(line_strip)
    # 扫尾
    if current_title_lines and len(current_title_lines) > 1:
        chunks.append({
            "content": "\n".join(current_title_lines),
            "title": current_title,
            "file_title": file_title
        })

    if len(chunks)==0 and len(current_title_lines) > 0:
        chunks.append({
            "content":"\n".join(current_title_lines),
            "title":"default",
            "file_title": file_title
        })
    logger.info(f"完成了语义标题切割，切块数量为:{len(chunks)}")
    return chunks


def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分服务：
    1. 按标题层级做一级粗切
    2. 对超长文本做二次细切
    3. 构造 chunks 列表
    4. 回写 chunks
    """
    # 1.获取参数的校验
    md_content,file_title = load_markdown_content(state)

    # 2.确保语义切割，根据标题切割
    chunks:list[dict[str,Any]] = split_chunks_document(md_content,file_title)

    return state