import os
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Path
from typing import Any, Dict, List
from app.models.algorithm2.algorithm2_config import Algorithm2Config
from app.models.algorithm2.algorithm2_result import Algorithm2Result
from app.schemas.algorithm2.algorithm2_config import Algorithm2ConfigCreate
from app.schemas.algorithm2.algorithm2_result import Algorithm2ResultResponse
from app.utils.csv_importer import import_csv_to_model

router = APIRouter()


@router.get("/configs", response_model=int)
async def get_config_id(
    tree_count: int = Query(..., description="决策树数量", ge=1),
    max_depth: int = Query(..., description="树最大深度", ge=2),
    sensitivity: float = Query(..., description="偏离敏感度", gt=0.0)
):
    """
    获取配置ID - 根据算法参数查找配置
    """
    config = await Algorithm2Config.get_or_none(
        tree_count=tree_count,
        max_depth=max_depth,
        sensitivity=sensitivity
    )

    if not config:
        raise HTTPException(status_code=404, detail="未找到配置")
    return config.config_id


@router.get("/configs/{config_id}/results", response_model=List[Algorithm2ResultResponse])
async def get_results(
    config_id: int = Path(..., description="配置ID"),
    area_code: str = Query(None, description="区域编码")
):
    """
    获取特定配置的算法结果
    可选参数：area_code 指定区域的结果
    """
    # 构建查询条件
    query_filters = {"config_id": config_id}
    if area_code:
        query_filters["area_code"] = area_code

    results = await Algorithm2Result.filter(**query_filters)

    if not results:
        error_detail = f"未找到配置ID为{config_id}的结果"
        if area_code:
            error_detail += f"，区域编码：{area_code}"
        raise HTTPException(status_code=404, detail=error_detail)
    return results


@router.get("/results", response_model=List[Algorithm2ResultResponse])
async def get_results_by_params(
    tree_count: int = Query(..., description="决策树数量", ge=1),
    max_depth: int = Query(..., description="树最大深度", ge=2),
    sensitivity: float = Query(..., description="偏离敏感度", gt=0.0),
    area_code: str = Query(None, description="区域编码")
):
    """
    获取算法结果 - 根据算法参数组合查询
    """
    # 查找匹配参数的配置
    config = await Algorithm2Config.get_or_none(
        tree_count=tree_count,
        max_depth=max_depth,
        sensitivity=sensitivity
    )

    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"未找到配置：tree_count={tree_count}, max_depth={max_depth}, sensitivity={sensitivity}"
        )

    # 使用找到的配置ID查询结果
    query_filters = {"config_id": config.config_id}
    if area_code:
        query_filters["area_code"] = area_code

    results = await Algorithm2Result.filter(**query_filters)

    if not results:
        error_detail = f"未找到配置ID为{config.config_id}的结果"
        if area_code:
            error_detail += f"，区域编码：{area_code}"
        raise HTTPException(status_code=404, detail=error_detail)

    return results


router.post("/configs/import-results", response_model=Dict[str, Any])


async def create_config_and_import_results(
    config: Algorithm2ConfigCreate,
    background_tasks: BackgroundTasks
):
    """
    创建或查找配置并导入CSV数据

    1. 根据传入的配置参数查找或创建配置
    2. 将CSV数据导入到algorithm2_result表中
    """
    # 1. 查找或创建配置
    db_config = await Algorithm2Config.get_or_none(
        tree_count=config.tree_count,
        max_depth=config.max_depth,
        sensitivity=config.sensitivity
    )

    # 如果配置不存在，创建新配置
    if not db_config:
        db_config = await Algorithm2Config.create(
            tree_count=config.tree_count,
            max_depth=config.max_depth,
            sensitivity=config.sensitivity
        )
        action = "created"
    else:
        action = "found"

    # 2. 构建CSV文件的完整路径
    # 获取项目根目录，当前文件在app/api/endpoints
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    csv_path = os.path.join(project_root, 'app', 'data',
                            'algorithm2', 'monitoring_points_weights.csv')

    # 3. 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=404,
            detail=f"CSV文件不存在: {csv_path}"
        )

    # 4. 导入CSV数据
    # 使用后台任务异步导入，避免阻塞API响应
    background_tasks.add_task(
        import_csv_to_model,
        csv_path=csv_path,
        model_class=Algorithm2Result,
        config_id=db_config.config_id,
        config_field_name="config_id",
        batch_size=100,
        encoding='utf-8'
    )

    return {
        "message": "已开始导入数据",
        "status": "processing",
        "config": {
            "config_id": db_config.config_id,
            "tree_count": db_config.tree_count,
            "max_depth": db_config.max_depth,
            "sensitivity": db_config.sensitivity,
            "action": action
        },
        "csv_path": csv_path
    }
