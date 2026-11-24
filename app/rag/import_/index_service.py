from pymilvus import DataType
from app.shared.runtime.logger import logger, step_log
from app.process.import_.agent.state import ImportGraphState
from app.infra.vector_store.milvus_gateway import milvus_gateway


@step_log("require_embeddings_content")
def require_embeddings_content(state: dict) -> list[dict]:
    """
    校验导入状态中是否已经生成切块结果
    功能：确保后续流程有有效的输入数据，缺失时抛出异常
    :param state: LangGraph 流程状态字典
    :return: 已通过校验的切块列表
    """
    # 从 state 中获取核心数据
    embeddings_content = state.get("embeddings_content", [])
    # ===================== 校验 chunks =====================
    # 如果 embeddings_content 为空，无法继续业务，直接抛出异常终止流程
    if not embeddings_content:
        logger.error("embeddings_content为空,无法继续业务!!")
        raise ValueError("embeddings_content为空,无法继续业务!!")

    # 返回校验后的数据
    return embeddings_content


def prepare_chunks_collection():
    milvus_client = milvus_gateway.milvus_client
    collection_name = milvus_gateway.chunk_collection_name
    if milvus_client.has_collection(collection_name=collection_name):
        return
    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)

    # 添加主键字段：chunk_id，INT64 类型，自增
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)

    # 添加文件标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)

    # 添加主体名称字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)

    # 添加切片标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)

    # 添加父标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)

    # 添加切片序号字段：INT8 类型
    schema.add_field(field_name="part", datatype=DataType.INT8)

    # 添加内容字段：VARCHAR 类型，最大长度 65535（支持长文本）
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)

    # 添加稠密向量字段：FLOAT_VECTOR 类型，维度 1024
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)

    # 添加稀疏向量字段：SPARSE_FLOAT_VECTOR 类型
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # ===================== 创建索引 =====================
    # 准备索引参数
    index_params = milvus_client.prepare_index_params()

    # 为稠密向量创建索引：使用 AUTOINDEX，metric_type 为 IP（内积）
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        index_name="dense_vector_index",
        metric_type="COSINE",
        params={
            "M": 64,  # 每个点最大的链接数量
            "efConstruction": 100  # 考虑链接的范围 在这个范围内确定 M
        }
    )

    # 为稀疏向量创建索引：使用 SPARSE_INVERTED_INDEX，算法为 DAAT_MAXSCORE
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )

    # 创建集合并应用索引
    milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


@step_log("remove_old_chunks")
def remove_old_chunks(file_title: str) -> None:
    """
    根据主体名称删除已存在的切片记录
    功能：实现幂等性，确保同一主体重复导入时覆盖旧数据
    :param item_name: 主体名称
    :return: 无返回值
    """
    # 获取 Milvus 客户端并执行删除操作
    milvus_gateway.milvus_client.delete(
        collection_name=milvus_gateway.chunk_collection_name,
        filter=f"file_title=='{file_title}'",
    )


@step_log("insert_embeddings_content")
def insert_embeddings_content(embeddings_content: list[dict]) -> None:
    """
    批量插入切片数据到 Milvus 集合
    功能：将带向量的切片数据持久化存储到向量库
    :param embeddings_content: 带向量字段的切片列表
    :return: 无返回值
    """
    # 执行批量插入操作
    result = milvus_gateway.milvus_client.insert(
        collection_name=milvus_gateway.chunk_collection_name,
        data=embeddings_content,
    )

    # 记录插入结果
    logger.info(f"插入数据成功! 总条数:{result.get('insert_count', 0)}")
    logger.info(f"插入数据主键回显:{result.get('ids', [])}")


def index_chunks(state: ImportGraphState) -> ImportGraphState:
    """
    入库服务：
    1. 准备集合 schema 和索引
    2. 根据 item_name 删除旧数据
    3. 批量插入新的 chunks
    4. 回写 chunk_id 等入库结果
    """
    # 1.获取并校验参数embedding_content
    embeddings_content = require_embeddings_content(state)
    # 2.准备并创建Milvus集合（获取Milvus客户端 -> 检查是否存在，存在直接返回，不存在则创建 -> 插入）
    # chunk_id,title,file_title,parent_title,part,content,dense_vector,sparse_vector
    prepare_chunks_collection()
    # 3.删除和插入
    file_title = state.get("file_title", "")
    if file_title:
        remove_old_chunks(file_title)
    # 4.打印日志 项目导入模块全部完成
    insert_embeddings_content(embeddings_content)

    return state
