import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNK_SIZE, CHUNK_MIN, CHUNK_MAX_SIZE
from app.shared.runtime.logger import logger


def load_markdown_content(state: ImportGraphState) -> tuple[str, str]:
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

    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    state["md_content"] = md_content
    return md_content, file_title


def split_chunks_document(md_content, file_title) -> list[dict[str, Any]]:
    """
    根据标题完成语义切割
    :param md_content:
    :param file_title:
    :return:
    """
    md_content_lines: list[str] = md_content.split('\n')

    chunks: list[dict[str, Any]] = []  # [{title,content,file_title}]
    current_title: str | None = None
    current_title_lines: list[str] = []
    is_code_block = False
    title_reg = re.compile(r"^\s*#{1,6}\s.+")

    for line in md_content_lines:
        line_strip = line.strip()
        if not line_strip:  # 空行判断
            logger.warning("处理碰到空行!跳过本次处理!")
            continue
        if line_strip.startswith("```") or line_strip.startswith("~~~"): # 如果是代码块就跳过
            is_code_block = not is_code_block
            current_title_lines.append(line_strip)
            continue

        if not is_code_block and title_reg.match(line_strip):  # 如果遇到标题且不在代码块里
            if current_title and len(current_title_lines) > 1: # 如果当前块有内容，且标题也有内容（中间）
                chunks.append({     # 加入
                    "content": "\n".join(current_title_lines),
                    "title": current_title,
                    "file_title": file_title
                })
            if not current_title and len(current_title_lines) > 0: # 如果当前块有内容但是标题没内容
                current_title_lines.append(line_strip)
            else:
                current_title_lines = [line_strip]
            current_title = line_strip
        else:
            current_title_lines.append(line_strip)
    # 扫尾
    if current_title and len(current_title_lines) > 1:
        chunks.append({
            "content": "\n".join(current_title_lines),
            "title": current_title,
            "file_title": file_title
        })

    if len(chunks) == 0 and len(current_title_lines) > 0:
        chunks.append({
            "content": "\n".join(current_title_lines),
            "title": "default",
            "file_title": file_title
        })
    logger.info(f"完成了语义标题切割，切块数量为:{len(chunks)}")
    return chunks


def _split_long_chunk(chunk) -> list[dict[str, Any]]:
    """

    :param chunk: title file_title content
    :return:
    """
    content = chunk.get("content")
    title = chunk.get("title")
    file_title = chunk.get("file_title")
    clear_content = content[len(title) + 1:]

    sub_content_prefix = title + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE - len(sub_content_prefix),
        chunk_overlap=0,
        separators=["\n\n", "\n", "。 ", "! ", "? ", "; ", ".", "!", "?", ";", " "]
    )
    sub_chunk_list = []
    for part_index, split_text in enumerate(splitter.split_text(clear_content), start=1):
        split_text = split_text.strip()
        content_sub_new = sub_content_prefix + split_text
        sub_chunk_list.append({
            "title": f"{title}_第{part_index}部分",
            "content": content_sub_new,
            "file_title": file_title,
            "parent_title": title,
            "part": part_index
        })
    logger.info(f"进入标题:{title},完成切割后,切成:{len(sub_chunk_list)}块!")
    return sub_chunk_list


def _merge_short_chunks_same_parent_title(refine_list) -> list[dict[str, Any]]:
    """
    同一个标题下，过短（400）进行合并，不超过（1000）
    :param refine_list:
    :return:
    """
    merged_chunk_list = []
    # 1.定义一个合并的base chunk
    base_chunk = None  # 要合并的 不动 判断400
    # 2.循环处理chunK_list挪动要合并的元素
    for next_chunk in refine_list:
        if not base_chunk:
            base_chunk = next_chunk
            logger.info(f"短合并第一次进入，设置base_chunk内容!")
            continue
        # 3.合并的逻辑
        # base_chunk -> content < 400 条件1
        is_short_chunk = len(base_chunk.get("content")) < CHUNK_MIN
        # next_chunk -> parent_title == base_chunk -> parent_chunk adn parent_title #非空
        # 确保不同语义标题一定不能合并!!!

        is_same_parent_title = base_chunk.get("parent_title") and base_chunk.get("parent_title") == next_chunk.get(
            "parent_title")

        if is_short_chunk and is_same_parent_title:
            # 短 + 同一个标题
            # base_chunk + next_chunk content <= 1000
            next_content = next_chunk.get("content")[len(next_chunk["parent_title"]) + 1:]
            is_not_long = len(base_chunk.get("content")) + len(next_content) <= CHUNK_MAX_SIZE
            if is_not_long:
                base_chunk['content'] = base_chunk.get("content") + "\n" + next_content

            else:
                merged_chunk_list.append(base_chunk)
                base_chunk = next_chunk
        else:
            merged_chunk_list.append(base_chunk)
            base_chunk = next_chunk

    if base_chunk:
        merged_chunk_list.append(base_chunk)

    logger.info(f"进行短合并，合并之前:{len(refine_list)},合并之后:{len(merged_chunk_list)}")

    return merged_chunk_list

def refine_chunks(chunks) -> list[dict[str, Any]]:
    """
    进行精细切割!
      长 -> 600 -> 短切
      短 -> 400 -> 合并 -> 1000
    :param chunks:
    :return:
    """
    # 1. 定义接收最终结果的list
    refine_list = []
    # 2. 进行循环处理查看是否过长
    for chunk in chunks:
        content = chunk.get("content")
        if len(content) > CHUNK_SIZE:
            long_chunk_list = _split_long_chunk(chunk)
            refine_list.extend(long_chunk_list)
        else:
            refine_list.append(chunk)
        # 3. 进行短合并处理
    refine_list = _merge_short_chunks_same_parent_title(refine_list)

    # 4. 补全属性..
    for chunk in refine_list:
        if "parent_title" not in chunk:
            chunk['parent_title'] = chunk.get('title', "default_title")
        if "part" not in chunk:
            chunk['part'] = 1
    # 5. 最终返回结果
    logger.info(f"完成chunks的精细处理! 进入切块数量:{len(chunks)},处理后:{len(refine_list)}")
    return refine_list


def backup_chunks_json(refine_chunks_list, md_path:str):
    # 1.获取目标地址
    json_path_obj:Path = Path(md_path).with_name(f"{Path(md_path).stem}.json")
    # 2.写入
    json_path_obj.write_text(json.dumps(refine_chunks_list,indent=4,ensure_ascii=False),encoding="utf-8")


def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分服务：
    1. 按标题层级做一级粗切
    2. 对超长文本做二次细切
    3. 构造 chunks 列表
    4. 回写 chunks
    """
    # 1.获取参数的校验
    md_content, file_title = load_markdown_content(state)

    # 2.确保语义切割，根据标题切割
    chunks: list[dict[str, Any]] = split_chunks_document(md_content, file_title)

    # 3. 进行精细切割处理
    refine_chunks_list = refine_chunks(chunks)

    # 4. 修改state -> chunks
    state['chunks'] = refine_chunks_list
    # 5. 备份refine_chunks_list  -> [{},{}] -> json字符串 -> 本地磁盘
    backup_chunks_json(refine_chunks_list, state['md_path'])

    return state
