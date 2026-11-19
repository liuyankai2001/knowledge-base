import re
from pathlib import Path

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import PORTED_IMAGE_EXTENSIONS
from app.shared.runtime.logger import logger


def validates_and_get_data(state: ImportGraphState) -> tuple[str, Path, Path]:
    md_path = state.get("md_path")

    if not md_path:
        logger.error(f"核心参数md_path为空，业务无法继续，提前终止!")
        raise ValueError(f"核心参数md_path为空,业务无法继续,提前终止!")
    md_path_obj = Path(md_path)
    if not md_path_obj.exists():
        logger.error(f"md_path地址为:{md_path},但是没有真实的文件，业务无法继续，提前终止!")
        raise FileNotFoundError(f"md_path地址为:{md_path},但是没有真实的文件，业务无法继续，提前终止!")

    md_content = md_path_obj.read_text(encoding='utf-8')
    if not md_content:
        logger.error(f"md_path地址为:{md_path}，有真实的文件,但是内容为空!业务无法继续，提前终止!")
        raise FileNotFoundError(
            f"md_path地址为:{md_path}，有真实的文件，但是内容为空!业务无法继续，提前终止!")
    state['md_content'] = md_content

    images_path_obj: Path = md_path_obj.parent / 'images'
    return md_content, images_path_obj, md_path_obj


def scan_images(images_path_obj: Path, md_content: str) -> list[tuple[str, str, tuple[str, str]]]:
    images_context = []
    for image_obj in images_path_obj.iterdir():
        image_name: str = image_obj.name
        if image_obj.suffix not in PORTED_IMAGE_EXTENSIONS:
            logger.warning(f"当前文件名：{image_name}，不是图片，无需处理，直接跳过本次!")
            continue
        image_reg = re.compile(r'\!\[.*?\]\(.*?' + re.escape(image_name) + r'.*?\)')
        match = image_reg.search(md_content)  # 找第一个匹配对象
        if not match:
            logger.warning(f"{image_name}图片没有被md内容引用，无需处理，直接跳过本次!")
            continue
        start = match.start()
        end = match.end()

        pre_context = md_content[max(0, start - 100):start]
        post_context = md_content[end:min(len(md_content), end + 100)]
        images_context.append(
            (
                image_name, str(image_obj),
                (pre_context, post_context)
            )
        )
    return images_context


def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    """
    Markdown 图片增强服务：
    1. 扫描 Markdown 中的图片
    2. 调用多模态模型生成图片说明
    3. 上传图片到 MinIO
    4. 替换 Markdown 图片地址并回写 md_content
    """
    # 1.参数校验和获取
    md_content, images_path_obj, md_path_obj = validates_and_get_data(state)

    # 2.判断是否有图片
    if (not images_path_obj.exists()) or (images_path_obj.is_file() and len(list(images_path_obj.iterdir())) == 0):
        logger.info(f"{md_path_obj}文档对应的images为空或者没有图片!不需要图片识别，提前结束当前节点!")
        return state
    # 3.获取每张图的信息
    # [（xx.png, c: // xx.png,（上文，下文））]
    image_content: list[tuple[str, str, tuple[str, str]]] = scan_images(images_path_obj, md_content)

    return state
