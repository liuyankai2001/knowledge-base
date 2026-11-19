# 引入现有的功能进行汇总
from minio import Minio
from app.shared.clients.minio_utils import get_minio_client
from app.infra.config.providers import infra_config

class MinioGateWay:

    # 提供获取桶名称的函数
    @property
    def bucket_name(self):
        return infra_config.minio_config.bucket_name

    # 提供获取图片前缀的函数
    @property
    def minio_img_dir(self):
        return infra_config.minio_config.minio_img_dir

    # 提供获取minio_client的函数
    @property
    def minio_client(self):
        return get_minio_client()


    def build_image_url(self,stem,object_name:str):
        protocol = "https://" if infra_config.minio_config.minio_secure else "http://"
        ip_add = infra_config.minio_config.endpoint
        bucket_name = infra_config.minio_config.bucket_name
        img_dir = infra_config.minio_config.minio_img_dir
        image_url = f"{protocol}{ip_add}/{bucket_name}{img_dir}/{stem}/{object_name}"
        return image_url

minio_gateway = MinioGateWay()