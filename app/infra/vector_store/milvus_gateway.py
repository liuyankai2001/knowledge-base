from app.infra.config.providers import infra_config
from app.shared.clients import get_milvus_client


class MilvusGateway:
    # 提供2个属性
    @property
    def chunk_collection_name(self):
        return infra_config.milvus_config.chunks_collection

    @property
    def item_name_collection_name(self):
        return infra_config.milvus_config.item_name_collection

    @property
    def milvus_client(self):
        return get_milvus_client()

    # def embedding_dim(self):
    #     return infra_config.milvus_config.

milvus_gateway = MilvusGateway()