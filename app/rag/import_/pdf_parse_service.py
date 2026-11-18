import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import PDF_PARSE_SERVICE_LOCAL_DIR, MINERU_MODEL_VERSION, MINERU_POLL_INTERVAL_SECONDS, \
    MINERU_POLL_TIMEOUT_SECONDS
from app.shared.runtime.logger import logger, PROJECT_ROOT
from app.infra.config.providers import infra_config


def validate_pdf_paths(state: ImportGraphState) -> tuple[Path, Path]:
    pdf_path = state.get('pdf_path')
    local_dir = state.get('local_dir')

    if not pdf_path:
        logger.error(f"pdf_path为空!")
        raise ValueError(f"pdf_path为空!")
    if not local_dir:
        logger.warning(f"local_dir为空,给予默认值,默认为:项目跟地址/output 文件夹")
        local_dir = PROJECT_ROOT / PDF_PARSE_SERVICE_LOCAL_DIR
        state["local_dir"] = str(local_dir)

    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    if not pdf_path_obj.exists():
        logger.error(f"{pdf_path}地址pdf文件不存在!")
        raise FileNotFoundError(f"{pdf_path}地址pdf文件不存在!")

    if not local_dir_obj.is_dir():
        logger.warning(f"{local_dir}地址目录不存在!")
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, local_dir_obj


def upload_pdf_and_poll(pdf_path_obj) -> str:
    header = {
        "Content-Type": "application/json",
        # .env配置文件 -> config / minerU_config / infra / config / providers / miner_u...
        "Authorization": f"Bearer {infra_config.mineru_config.api_key}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}"}
        ],
        "model_version": MINERU_MODEL_VERSION
    }
    url = f"{infra_config.mineru_config.base_url}/file-urls/batch"

    response = requests.post(url=url, headers=header, json=data)
    if response.status_code != 200:
        logger.error(f"向mineru服务器申请上传文件解析，但是http状态码为{response.status_code},状态错误,无法继续!")
        raise RuntimeError(f"向mineru服务器申请上传文件解析，但是http状态码为{response.status_code},状态错误,无法继续!")

    response_dict = response.json()
    if response_dict.get('code', -1) != 0:
        logger.error(f"向mineru服务器申请上传文件解析,网络正常,服务业务状态异常 code={response_dict.get('code', -1)}"
                     f"错误原因:{response_dict.get('msg')},无法继续业务")
        raise RuntimeError(
            f"向mineru服务器申请上传文件解析,网络正常,服务业务状态异常 code={response_dict.get('code', -1)}"
            f"错误原因:{response_dict.get('msg')},无法继续业务")
    batch_id = response_dict.get('data', {}).get('batch_id')

    file_upload_urls = response_dict.get('data', {}).get('file_urls', [])
    file_upload_url = None
    if len(file_upload_urls) > 0:
        file_upload_url = file_upload_urls[0]
    if not batch_id:
        logger.error("向mineru申请解析文件，返回的batch_id为空，业务中断")
        raise ValueError("向mineru申请解析文件，返回的batch_id为空，业务中断")

    # 向指定的url发起网络请求并上传地址
    with requests.Session() as session:  # 按照代码方式直接传递参数，不额外添加请求内容(预签名)
        session.trust_env = False
        upload_response = requests.put(url=file_upload_url, data=pdf_path_obj.read_bytes())

    if upload_response.status_code != 200:
        logger.error(f"向{file_upload_url}上传文件，服务器返回的网络状态码为：{upload_response.status_code}")
        raise RuntimeError(f"向{file_upload_url}上传文件，服务器返回的网络状态码为：{upload_response.status_code}")
    logger.info(f"完成上传文件申请，batch_id:{batch_id},上传文件预签名地址:{file_upload_url}")

    # 轮询向mineru获取batch_id解析状态zip_url
    result_url = f"{infra_config.mineru_config.base_url}/extract-results/batch/{batch_id}"
    start_time = time.time()
    while True:
        if time.time() - start_time >= MINERU_POLL_TIMEOUT_SECONDS:
            logger.error(f"轮询获取batch_id:{batch_id}对应的解析结果超时，耗时:{time.time() - start_time}")
            raise TimeoutError(f"轮询获取batch_id:{batch_id}对应的解析结果超时，耗时:{time.time() - start_time}")
        try:
            poll_result = requests.get(result_url, headers=header)
        except Exception as e:
            logger.warning(f"申请结果出现网络波动{str(e)}，稍后再试!")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue
        # 网络状态判定
        # 1 2 3 4 5
        if poll_result.status_code != 200:
            if 500 <= poll_result.status_code < 600:
                logger.warning(f"申请结果出现网络波动{poll_result.status_code}，稍后再试,等待服务器修复!")
                time.sleep(MINERU_POLL_INTERVAL_SECONDS)
                continue
            else:
                logger.error(
                    f"获取{batch_id}对应的解析结果，服务器访问报错，http的状态码:{poll_result.status_code},错误无法修复")
                raise RuntimeError(
                    f"获取{batch_id}对应的解析结果，服务器访问报错，http的状态码:{poll_result.status_code},错误无法修复")

        # 业务状态判定
        poll_result_dict = poll_result.json()
        if poll_result_dict.get('code', -1) != 0:
            logger.error(
                f"获取{batch_id}对应的解析结果，业务状态报错，业务的状态码:{poll_result_dict.get('code', -1)},错误信息:{poll_result_dict.get('msg')},错误无法修复")
            raise RuntimeError(
                f"获取{batch_id}对应的解析结果，业务状态报错，业务的状态码:{poll_result_dict.get('code', -1)},错误信息:{poll_result_dict.get('msg')},错误无法修复")
        extract_result_list = poll_result_dict.get('data', {}).get('extract_result', [])
        if len(extract_result_list) == 0:
            # 错误是可以给机会！这次不行了
            logger.warning(f"解析结果extract_result_list为空,跳过本次!稍后再试")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue
        extract_result = extract_result_list[0]
        state = extract_result.get('state')
        if state == 'done':
            # 解析完毕
            full_zip_url = extract_result.get('full_zip_url')
            if not full_zip_url:
                logger.error(f"获取:{batch_id}对应的解析结果，任务已经完成，但是full_zip_url没有地址!业务失败，提前终止!")
                raise ValueError(f"获取:{batch_id}对应的解析结果，任务已经完成，但是full_zip_url没有地址!业务失败，提前终止!")
            return full_zip_url

        elif state == 'failed':
            # 解析完毕，失败了
            logger.error(f"获取:{batch_id}对应的解析结果，任务解析失败!业务失败，提前终止!")
            raise ValueError(f"获取:{batch_id}对应的解析结果，任务解析失败!业务失败，提前终止!")

        else:
            # 正在解析中...
            logger.warning("本次解析，没有获得结果，继续下一次!!!")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue



def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """
    PDF 解析服务：
    1. 调用 MinerU
    2. 下载并解压解析结果
    3. 获取 Markdown 路径和正文内容
    4. 回写 md_path / md_content / local_dir
    """
    # 1.获取并校验参数
    pdf_path_obj, local_dir_obj = validate_pdf_paths(state)
    # 2.mineru解析pdf文件并返回zip的下载地址
    zip_url: str = upload_pdf_and_poll(pdf_path_obj)

    return state
