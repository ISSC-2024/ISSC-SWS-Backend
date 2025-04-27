import os
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Path, Depends
from typing import Any, Dict, List, Optional
from app.models.algorithm3.algorithm3_config import Algorithm3Config
from app.models.algorithm3.algorithm3_result import Algorithm3Result
from app.schemas.algorithm3.algorithm3_config import Algorithm3ConfigBase, Algorithm3ConfigCreate, Algorithm3ConfigResponse
from app.schemas.algorithm3.algorithm3_result import Algorithm3ResultResponse
from app.utils.csv_importer import import_csv_to_model

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


@router.get("/configs", response_model=int)
async def get_config_id(
    config_params: Algorithm3ConfigBase = Depends()
):
    """
    获取配置ID - 根据算法参数查找配置
    """
    config = await get_config_from_params(config_params)

    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"未找到配置：algorithm={config_params.algorithm}, learning_rate={config_params.learning_rate}"
        )

    return config.config_id


@router.get("/configs/{config_id}/results", response_model=List[Algorithm3ResultResponse])
async def get_results(
    config_id: int = Path(..., description="配置ID"),
    point_id: str = Query(None, description="监测点ID")
):
    """
    获取特定配置的算法结果
    可选参数：point_id 指定监测点的结果
    """
    # 构建查询条件
    query_filters = {"config_id": config_id}
    if point_id:
        query_filters["point_id"] = point_id

    results = await Algorithm3Result.filter(**query_filters)

    if not results:
        error_detail = f"未找到配置ID为{config_id}的结果"
        if point_id:
            error_detail += f"，监测点ID：{point_id}"
        raise HTTPException(status_code=404, detail=error_detail)
    return results


@router.get("/results", response_model=List[Algorithm3ResultResponse])
async def get_results_by_params(
    config_params: Algorithm3ConfigBase = Depends(),
    point_id: str = Query(None, description="监测点ID")
):
    """
    获取算法结果 - 根据算法参数组合查询
    """
    # 查找匹配参数的配置
    config = await get_config_from_params(config_params)

    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"未找到配置：algorithm={config_params.algorithm}, learning_rate={config_params.learning_rate}"
        )

    # 使用找到的配置ID查询结果
    query_filters = {"config_id": config.config_id}
    if point_id:
        query_filters["point_id"] = point_id

    results = await Algorithm3Result.filter(**query_filters)

    if not results:
        error_detail = f"未找到配置ID为{config.config_id}的结果"
        if point_id:
            error_detail += f"，监测点ID：{point_id}"
        raise HTTPException(status_code=404, detail=error_detail)

    return results

# 创建新配置


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
    existing_config = await Algorithm3Config.get_or_none(**create_params)
    if not existing_config:
        # db_config = await Algorithm3Config.create(**create_params)
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
    # 将config转换为ConfigBase对象，用于查询
    config_params = Algorithm3ConfigBase(
        algorithm=config.algorithm,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        max_epochs=config.max_epochs
    )

    # 1. 查找现有配置
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

    # 2. 构建CSV文件的完整路径
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

    # 完整的CSV文件路径
    csv_path = os.path.join(
        project_root,
        'app',
        'data',
        'algorithm3',
        algorithm_name,  # 算法名子文件夹
        model_dir_name,  # 模型子文件夹
        csv_name         # CSV文件名
    )

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
