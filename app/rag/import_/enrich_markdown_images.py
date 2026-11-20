import re
import base64

from pathlib import Path
from mimetypes import guess_type
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import PORTED_IMAGE_EXTENSIONS
from app.shared.runtime.logger import logger,step_log
from app.infra.llm.providers import llm_providers
from app.infra.object_storage.minio_gateway import minio_gateway
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.rate_limit_utils import apply_api_rate_limit


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


def summarize_images(image_content: list[tuple[str, str, tuple[str, str]]], stem: str) -> dict[str, str]:
    """
    :param image_content:
    :param stem:
    :return:
    """
    image_summaries = {}

    # 7.1 获取模型对象 vision_chat()
    vision_client = llm_providers.vision_chat()

    # 循环处理每一张图片
    for image_name, image_path, context in image_content:
        # 添加访问限制
        apply_api_rate_limit()
        # 7.2 封装提示词（图片 / 文本）
        image_text = load_prompt("image_summary", root_folder=stem, image_content=context)
        image_path_obj: Path = Path(image_path)
        image_base64_str = base64.b64encode(image_path_obj.read_bytes()).decode(encoding="utf-8")

        human_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": image_text
                },

                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{guess_type(image_name)[0]};base64,{image_base64_str}"
                    }
                },

            ]
        )

        # 7.3 封装一个调用chains StrOutputParser()
        chains = vision_client | StrOutputParser()
        # 7.4 执行调用链获取返回结果
        image_summary = chains.invoke([human_message])
        image_summaries[image_name] = image_summary
        logger.debug(f"完成:{image_name}图片意图识别,识别内容:{image_summary}")
    # 7.5 封装一个dict，并且返回即可
    return image_summaries


def upload_images_and_replace(md_content: str, image_summaries: dict[str, str],
                              image_content: list[tuple[str, str, tuple[str, str]]], stem: str) -> str:
    """
    进行文件上传和md_content替换
    :param md_content:旧的md_content
    :param image_summaries:图片的总结描述
    :param image_content:
    :param stem:
    :return:
    """
    minio_client = minio_gateway.minio_client
    # 先删除旧的
    select_object_list = minio_client.list_objects(
        bucket_name=minio_gateway.bucket_name,
        prefix=minio_gateway.minio_img_dir[1:] + '/' + stem,
        recursive=True  # 获取前缀文件夹下所有对象
    )
    delete_object_list = [DeleteObject(select_object.object_name) for select_object in select_object_list]
    errors = minio_client.remove_objects(bucket_name=minio_gateway.bucket_name, delete_object_list=delete_object_list)
    for error in errors:
        logger.warning(f"文件删除状态:{error}")
    # 上传
    image_urls = {}
    for image_name, image_path_str, _ in image_content:
        try:
            object_name = minio_gateway.minio_img_dir + '/' + stem + '/' + image_name
            minio_client.fput_object(bucket_name=minio_gateway.bucket_name, object_name=object_name,
                                     file_path=image_path_str,
                                     content_type=guess_type(image_name)[0])
            # 图片访问地址
            # http://ip:9000/桶名/文件夹名/图片名
            image_url = minio_gateway.build_image_url(stem, image_name)
            image_urls[image_name] = image_url

        except Exception as e:
            logger.warning(f"{image_name}图片上传minio失败，原因:{e},跳过本次，继续运行!")
            continue
    if not image_urls:
        logger.warning(f"图片上传全部失败!直接使用原md_content处理即可!")
        return md_content

    # 进行md_content内容替换
    for image_name, image_url in image_urls.items():
        image_summary = image_summaries.get(image_name)
        reg = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        md_content = reg.sub(lambda _: f"![{image_summary}]({image_url})", md_content)
    return md_content


def backup_new_md_content(md_content_new: str, md_path_obj: Path) -> str:
    """
    备份
    :param md_content_new:
    :param md_path_obj:
    :return:
    """
    md_new_path_obj = md_path_obj.with_name(f"{md_path_obj.stem}_new{md_path_obj.suffix}")
    md_new_path_obj.write_text(md_content_new, encoding='utf-8')
    logger.info(f"将md_content_new进行备份，备份地址为：{md_new_path_obj}")
    return str(md_new_path_obj)

@step_log("enrich_markdown_images")
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

    # 4.进行图片内容识别(视觉模型)
    image_summarizes: dict[str, str] = summarize_images(image_content, md_path_obj.stem)

    # 5.文件上传和md_content内容替换
    md_content_new: str = upload_images_and_replace(md_content, image_summarizes, image_content, md_path_obj.stem)

    # 6.修改state中的md_content
    state['md_content'] = md_content_new

    # 7.备份md_content
    md_path_new: str = backup_new_md_content(md_content_new, md_path_obj)
    state['md_path'] = md_path_new

    return state
