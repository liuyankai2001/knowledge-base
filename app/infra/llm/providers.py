from app.infra.config.providers import infra_config
from app.shared.model import get_bge_m3_ef, generate_embeddings
from app.shared.model.lm_utils import get_llm_client


class LLMProvider:

    # 获取普通大语言模型
    def chat(self,model_name:str=None,json_mode:bool=None):
        return get_llm_client(model=model_name,json_mode=json_mode)

    def vision_chat(self,vision_mode_name:str = None):
        return get_llm_client(vision_mode_name or infra_config.lm_config.lv_model)

    def bge_m3_embedding(self):
        return get_bge_m3_ef()

    def generate_embeddings(self,texts:list[str]):
        return generate_embeddings(texts)


llm_providers = LLMProvider()