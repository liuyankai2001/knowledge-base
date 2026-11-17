from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger , step_log
from pathlib import Path

@step_log("resolve_input_file")
def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    """
     进行文件类型调用分发任务
    :param state:
    :return:
    """
    return state