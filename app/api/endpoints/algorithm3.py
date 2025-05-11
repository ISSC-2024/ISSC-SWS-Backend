import os
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Path, Depends
from typing import Any, Dict, List, Optional, Tuple

from fastapi.responses import StreamingResponse

from app.models.algorithm3.algorithm3_config import Algorithm3Config
from app.models.algorithm3.algorithm3_result import Algorithm3Result
from app.schemas.algorithm3.algorithm3_config import Algorithm3ConfigBase, Algorithm3ConfigCreate, Algorithm3ConfigResponse
from app.schemas.algorithm3.algorithm3_result import PaginatedAlgorithm3Results
from app.utils.csv_builder import CSVBuilder
from app.utils.csv_importer import import_csv_to_model
from app.utils.json_importer import import_json_to_model

router = APIRouter()


async def get_config_from_params(config_params: Algorithm3ConfigBase) -> Optional[Algorithm3Config]:
    """
    根据配置参数获取对应的配置对象

    返回:
        Algorithm3Config 或 None (如果未找到配置)
    """
    # 构建查询条件
    query_params = {
        "algorithm": config_params.algorithm,
        "learning_rate": config_params.learning_rate
    }

    # 根据算法类型添加特定参数
    if config_params.algorithm in ['xgboost', 'lightGBM']:
        query_params["max_depth"] = config_params.max_depth
    elif config_params.algorithm == 'TabNet':
        query_params["max_epochs"] = config_params.max_epochs

    return await Algorithm3Config.get_or_none(**query_params)


async def get_existing_config(config_params: Algorithm3ConfigBase) -> Algorithm3Config:
    """
    获取已存在的配置，如果不存在则抛出404异常
    """
    config = await get_config_from_params(config_params)
    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"未找到配置：algorithm={config_params.algorithm}, learning_rate={config_params.learning_rate}"
        )
    return config


# 查询结果的通用函数，支持分页
async def get_results_with_filters(
    config_id: int,
    region: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    get_all: bool = False
) -> Tuple[List[Algorithm3Result], int]:
    """
    根据配置ID和可选的区域过滤器获取结果，支持分页或获取全部数据

    参数:
        config_id: 配置ID
        region: 可选的区域过滤条件
        skip: 跳过的记录数
        limit: 返回的最大记录数
        get_all: 是否返回所有符合条件的数据，为True时忽略skip和limit参数

    返回:
        Tuple[List[Algorithm3Result], int]: (结果列表, 总记录数)
    """
    # 构建查询条件
    query_filters = {"config_id": config_id}
    if region:
        query_filters["region"] = region

    # 首先获取总记录数
    total_count = await Algorithm3Result.filter(**query_filters).count()

    # 如果没有记录，抛出404错误
    if total_count == 0:
        error_detail = f"未找到配置ID为{config_id}的结果"
        if region:
            error_detail += f"，区域：{region}"
        raise HTTPException(status_code=404, detail=error_detail)

    # 根据get_all参数决定是否应用分页
    if get_all:
        results = await Algorithm3Result.filter(**query_filters).all()
    else:
        results = await Algorithm3Result.filter(**query_filters).offset(skip).limit(limit)

    return results, total_count


def get_model_path_info(config: Algorithm3ConfigCreate, file_name: str) -> Tuple[str, str]:
    """
    生成模型路径相关信息

    返回:
        Tuple[str, str]: (完整文件路径, 模型目录名)
    """
    # 获取项目根目录，当前文件在app/api/endpoints
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

    # 根据算法类型确定子文件夹路径
    algorithm_name = config.algorithm
    learning_rate = config.learning_rate

    # 确定使用max_depth还是max_epochs
    if algorithm_name in ['xgboost', 'lightGBM']:
        model_param = config.max_depth
    else:  # TabNet
        model_param = config.max_epochs

    # 构建模型子目录名称
    model_dir_name = f"{algorithm_name}_model_{learning_rate}_{model_param}"

    # 完整的文件路径
    file_path = os.path.join(
        project_root,
        'app',
        'data',
        'algorithm3',
        algorithm_name,  # 算法名子文件夹
        model_dir_name,  # 模型子文件夹
        file_name        # 文件名
    )

    return file_path, model_dir_name


async def create_or_get_config(config: Algorithm3ConfigCreate) -> Tuple[Algorithm3Config, str]:
    """
    创建新配置或获取已存在的配置

    返回:
        Tuple[Algorithm3Config, str]: (配置对象, 动作[created/found])
    """
    # 将config转换为ConfigBase对象，用于查询
    config_params = Algorithm3ConfigBase(
        algorithm=config.algorithm,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        max_epochs=config.max_epochs
    )

    # 查找现有配置
    db_config = await get_config_from_params(config_params)

    # 如果配置不存在，创建新配置
    if not db_config:
        # 构建创建参数
        create_params = {
            "algorithm": config.algorithm,
            "learning_rate": config.learning_rate
        }

        if config.algorithm in ['xgboost', 'lightGBM']:
            create_params["max_depth"] = config.max_depth
        elif config.algorithm == 'TabNet':
            create_params["max_epochs"] = config.max_epochs

        try:
            db_config = await Algorithm3Config.create(**create_params)
            action = "created"
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        action = "found"

    return db_config, action


@router.get("/configs", response_model=int)
async def get_config_id(
    config_params: Algorithm3ConfigBase = Depends()
):
    """
    获取配置ID - 根据算法参数查找配置
    """
    config = await get_existing_config(config_params)
    return config.config_id


@router.get("/configs/{config_id}/results", response_model=PaginatedAlgorithm3Results)
async def get_results(
    config_id: int = Path(..., description="配置ID"),
    region: str = Query(None, description="区域"),
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(100, ge=1, le=500, description="返回的最大记录数"),
    get_all: bool = Query(False, description="是否返回所有符合条件的数据，为true时忽略分页参数")
):
    """
    获取特定配置的算法结果，支持分页或获取全部数据

    参数:
        config_id: 配置ID
        region: 可选的区域过滤
        skip: 分页偏移量，默认0
        limit: 每页记录数，默认100，最大500
        get_all: 设置为true时返回所有符合条件的数据

    返回:
        包含分页信息和结果数据的响应
    """
    results, total_count = await get_results_with_filters(config_id, region, skip, limit, get_all)

    # 构建分页信息
    pagination = {
        "total": total_count,
        "skip": 0 if get_all else skip,
        "limit": total_count if get_all else limit,
        "has_more": False if get_all else (skip + limit) < total_count
    }

    return {
        "pagination": pagination,
        "data": results
    }


@router.get("/results", response_model=PaginatedAlgorithm3Results)
async def get_results_by_params(
    config_params: Algorithm3ConfigBase = Depends(),
    region: str = Query(None, description="区域"),
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(100, ge=1, le=500, description="返回的最大记录数"),
    get_all: bool = Query(False, description="是否返回所有符合条件的数据，为true时忽略分页参数")
):
    """
    获取算法结果 - 根据算法参数组合查询，支持分页或获取全部数据

    参数:
        config_params: 算法配置参数
        region: 可选的区域过滤
        skip: 分页偏移量，默认0
        limit: 每页记录数，默认100，最大500
        get_all: 设置为true时返回所有符合条件的数据

    返回:
        包含分页信息和结果数据的响应
    """
    # 查找匹配参数的配置
    config = await get_existing_config(config_params)

    # 使用配置ID查询结果
    results, total_count = await get_results_with_filters(config.config_id, region, skip, limit, get_all)

    # 构建分页信息
    pagination = {
        "total": total_count,
        "skip": 0 if get_all else skip,
        "limit": total_count if get_all else limit,
        "has_more": False if get_all else (skip + limit) < total_count
    }

    return {
        "pagination": pagination,
        "data": results
    }


@router.get("/results/download-csv", response_class=StreamingResponse)
async def download_results_csv(
    config_params: Algorithm3ConfigBase = Depends(),
    region: Optional[str] = Query(None, description="区域过滤"),
    filename: Optional[str] = Query(None, description="自定义文件名前缀"),
    localize: bool = Query(False, description="是否使用中文列名")
):
    """
    下载算法结果为CSV格式（流式下载）

    参数:
        config_params: 算法配置参数 (algorithm, learning_rate等)
        region: 可选的区域过滤
        filename: 自定义文件名前缀
        localize: 使用中文列名

    返回:
        StreamingResponse: 流式CSV下载响应
    """
    try:

        # 1. 查找配置信息
        config = await get_existing_config(config_params)

        # 2. 查询符合条件的所有数据
        query_filters = {"config_id": config.config_id}
        if region:
            query_filters["region"] = region

        results = await Algorithm3Result.filter(**query_filters).all()

        # 3. 判断是否有数据
        if not results:
            return CSVBuilder.create_error_response(
                message=f"未找到符合条件的数据: 算法={config.algorithm}, 学习率={config.learning_rate}, 区域={region or '所有'}",
                filename_prefix="no_data"
            )

        # 4. 生成文件名前缀
        if not filename:
            # 根据算法参数构建有意义的文件名
            parts = [config.algorithm]

            # 添加学习率
            parts.append(f"lr{config.learning_rate}")

            # 根据算法类型添加特定参数
            if config.algorithm in ['xgboost', 'lightGBM'] and config.max_depth:
                parts.append(f"depth{config.max_depth}")
            elif config.algorithm == 'TabNet' and config.max_epochs:
                parts.append(f"epochs{config.max_epochs}")

            # 添加区域信息
            if region:
                parts.append(f"region_{region}")

            file_prefix = "_".join(parts)
        else:
            file_prefix = filename

        # 5. 设置列映射
        column_mapping = None
        if localize:
            column_mapping = {
                'timestamp': '时间戳',
                'region': '区域',
                'risk_level': '风险等级',
                'message': '消息'
            }

        # 6. 构建并返回CSV流式响应
        return CSVBuilder.create_streaming_response(
            results=results,
            filename_prefix=file_prefix,
            column_mapping=column_mapping,
            date_columns=['timestamp'],
            exclude_columns=['result_id', 'config_id']  # 排除不需要的字段
        )

    except HTTPException as e:
        # 处理找不到配置等错误
        return CSVBuilder.create_error_response(
            message=f"请求错误: {e.detail}",
            filename_prefix=f"error_{config_params.algorithm}"
        )
    except Exception as e:
        # 处理其他未预期的错误
        return CSVBuilder.create_error_response(
            message=f"服务器错误: {str(e)}",
            filename_prefix="server_error"
        )


@router.post("/configs", response_model=Algorithm3ConfigResponse)
async def create_config(
    config: Algorithm3ConfigCreate
):
    """
    创建新配置
    """
    # 构建创建参数
    create_params = {
        "algorithm": config.algorithm,
        "learning_rate": config.learning_rate
    }

    if config.algorithm in ['xgboost', 'lightGBM']:
        create_params["max_depth"] = config.max_depth
    elif config.algorithm == 'TabNet':
        create_params["max_epochs"] = config.max_epochs

    # 查找现有配置
    existing_config = await get_config_from_params(
        Algorithm3ConfigBase(
            algorithm=config.algorithm,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            max_epochs=config.max_epochs
        )
    )

    if not existing_config:
        # 直接返回创建的配置对象
        return Algorithm3ConfigResponse(**create_params)

    raise HTTPException(
        status_code=400,
        detail=f"配置已存在：algorithm={config.algorithm}, learning_rate={config.learning_rate}"
    )


@router.post("/configs/import-results", response_model=Dict[str, Any])
async def create_config_and_import_results(
    config: Algorithm3ConfigCreate,
    background_tasks: BackgroundTasks,
    csv_name: str = Query("predicted_results.csv",
                          description="CSV文件名，默认为predicted_results.csv")
):
    """
    创建或查找配置并导入CSV数据

    1. 根据传入的配置参数查找或创建配置
    2. 将CSV数据导入到algorithm3_result表中

    CSV文件路径规则：
    app/data/algorithm3/{算法名}/{算法名}_model_{learning_rate}_{max_depth或max_epochs}/{csv_name}
    """
    # 1. 创建或获取配置
    db_config, action = await create_or_get_config(config)

    # 2. 构建CSV文件的完整路径
    csv_path, model_dir_name = get_model_path_info(config, csv_name)

    # 3. 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=404,
            detail=f"CSV文件不存在: {csv_path}"
        )

    # 4. 导入CSV数据(后台)
    background_tasks.add_task(
        import_csv_to_model,
        csv_path=csv_path,
        model_class=Algorithm3Result,
        config_id=db_config.config_id,
        config_field_name="config",
        batch_size=100,
        encoding='utf-8'
    )

    return {
        "message": "已开始导入数据",
        "status": "processing",
        "config": {
            "config_id": db_config.config_id,
            "algorithm": db_config.algorithm,
            "learning_rate": db_config.learning_rate,
            "max_depth": db_config.max_depth,
            "max_epochs": db_config.max_epochs,
            "action": action
        },
        "csv_path": csv_path
    }


@router.post("/configs/import-json-results", response_model=Dict[str, Any])
async def create_config_and_import_json_results(
    config: Algorithm3ConfigCreate,
    background_tasks: BackgroundTasks,
    json_name: str = Query("predicted_results.json",
                           description="JSON文件名，默认为predicted_results.json")
):
    """
    创建或查找配置并导入JSON数据

    1. 根据传入的配置参数查找或创建配置
    2. 将JSON数据导入到algorithm3_result表中

    JSON文件路径规则：
    app/data/algorithm3/{算法名}/{算法名}_model_{learning_rate}_{max_depth或max_epochs}/{json_name}
    """
    # 1. 创建或获取配置
    db_config, action = await create_or_get_config(config)

    # 2. 构建JSON文件的完整路径
    json_path, model_dir_name = get_model_path_info(config, json_name)

    # 3. 检查JSON文件是否存在
    if not os.path.exists(json_path):
        raise HTTPException(
            status_code=404,
            detail=f"JSON文件不存在: {json_path}"
        )

    # 4. 导入JSON数据(后台)
    background_tasks.add_task(
        import_json_to_model,
        json_path=json_path,
        model_class=Algorithm3Result,
        config_id=db_config.config_id,
        config_field_name="config",
        batch_size=100,
        encoding='utf-8'
    )

    return {
        "message": "已开始导入数据",
        "status": "processing",
        "config": {
            "config_id": db_config.config_id,
            "algorithm": db_config.algorithm,
            "learning_rate": db_config.learning_rate,
            "max_depth": db_config.max_depth,
            "max_epochs": db_config.max_epochs,
            "action": action
        },
        "json_path": json_path
    }
