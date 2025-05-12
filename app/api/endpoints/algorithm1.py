import os
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Depends
from typing import Any, Dict, Optional, List, Tuple

from app.models.algorithm1.predictions_arima_auto import PredictionsArimaAuto
from app.schemas.algorithm1.algorithm1_result import (
    PaginationInfo,
    PaginatedPredictionsArimaAutos
)
from app.utils.csv_builder import CSVBuilder
from fastapi.responses import StreamingResponse

from app.utils.json_importer import import_json_to_model

router = APIRouter()

# 查询结果的通用函数，支持分页


async def get_predictions_with_filters(
    region: Optional[str] = None,
    point_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    get_all: bool = False
) -> Tuple[List[PredictionsArimaAuto], int]:
    """
    根据过滤条件获取预测结果，支持分页或获取全部数据

    参数:
        region: 可选的区域过滤条件
        point_id: 可选的监测点ID过滤条件
        skip: 跳过的记录数
        limit: 返回的最大记录数
        get_all: 是否返回所有符合条件的数据，为True时忽略skip和limit参数

    返回:
        Tuple[List[PredictionsArimaAuto], int]: (结果列表, 总记录数)
    """
    # 构建查询条件
    query_filters = {}
    if region:
        query_filters["region"] = region
    if point_id:
        query_filters["point_id"] = point_id

    # 获取总记录数
    total_count = await PredictionsArimaAuto.filter(**query_filters).count()

    # 根据get_all参数决定是否应用分页
    if get_all:
        results = await PredictionsArimaAuto.filter(**query_filters).order_by("timestamp").all()
    else:
        results = await PredictionsArimaAuto.filter(**query_filters).order_by("timestamp").offset(skip).limit(limit)

    return results, total_count


@router.get(
    "/arima",
    response_model=PaginatedPredictionsArimaAutos,
    summary="获取ARIMA预测结果",
    description="获取ARIMA自动预测结果，支持按区域过滤和分页"
)
async def get_arima_predictions(
    region: Optional[str] = Query(None, description="区域代码，不提供则返回所有区域结果"),
    point_id: Optional[str] = Query(None, description="可选的监测点ID"),
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(100, ge=1, le=500, description="返回的最大记录数"),
    get_all: bool = Query(False, description="是否返回所有符合条件的数据，为true时忽略分页参数"),
):
    """
    获取ARIMA自动预测结果，支持按区域过滤和分页

    参数:
    - **region**: 可选，区域代码，不提供则返回所有区域结果
    - **point_id**: 可选，特定监测点ID
    - **skip**: 分页偏移量，默认为0
    - **limit**: 每页记录数，默认100，最大500
    - **get_all**: 设置为true时返回所有符合条件的数据，忽略分页参数

    返回:
    - 包含分页信息和结果数据的响应
    """
    # 获取结果和总数
    results, total_count = await get_predictions_with_filters(
        region, point_id,  skip, limit, get_all
    )

    # 构建分页信息
    pagination = PaginationInfo(
        total=total_count,
        skip=0 if get_all else skip,
        limit=total_count if get_all else limit,
        has_more=False if get_all else (skip + limit) < total_count
    )

    # 如果没有找到数据，返回空列表
    if not results:
        return PaginatedPredictionsArimaAutos(
            pagination=pagination,
            data=[]
        )

    return PaginatedPredictionsArimaAutos(
        pagination=pagination,
        data=results
    )


@router.get("/arima/results/download-csv", response_class=StreamingResponse)
async def download_arima_predictions_csv(
    region: Optional[str] = Query(None, description="区域过滤"),
    point_id: Optional[str] = Query(None, description="监测点ID"),
    filename: Optional[str] = Query(None, description="自定义文件名前缀"),
    localize: bool = Query(False, description="是否使用中文列名"),
):
    """
    下载ARIMA预测结果为CSV格式

    参数:
        region: 可选的区域过滤
        point_id: 可选的监测点ID
        filename: 自定义文件名前缀
        localize: 使用中文列名
        start_time: 可选的开始时间
        end_time: 可选的结束时间

    返回:
        StreamingResponse: 流式CSV下载响应
    """
    try:
        # 1. 构建查询条件
        query_filters = {}
        filter_desc = []

        if region:
            query_filters["region"] = region
            filter_desc.append(f"区域={region}")

        if point_id:
            query_filters["point_id"] = point_id
            filter_desc.append(f"监测点={point_id}")

        # 2. 查询所有符合条件的数据
        results = await PredictionsArimaAuto.filter(**query_filters).order_by("timestamp").all()

        # 3. 判断是否有数据
        if not results:
            return CSVBuilder.create_error_response(
                message=f"未找到符合条件的数据: {', '.join(filter_desc) if filter_desc else '所有记录'}",
                filename_prefix="no_data"
            )

        # 4. 生成文件名前缀
        if not filename:
            parts = ["arima"]

            if region:
                parts.append(f"region_{region}")

            if point_id:
                parts.append(f"point_{point_id}")

            file_prefix = "_".join(parts)
        else:
            file_prefix = filename

        # 5. 设置列映射
        column_mapping = None
        if localize:
            column_mapping = {
                'timestamp': '时间戳',
                'point_id': '监测点ID',
                'region': '区域',
                'temperature': '温度',
                'pressure': '压力',
                'flow_rate': '流速',
                'level': '液位',
                'gas_type': '气体类型',
                'gas_concentration': '气体浓度'
            }

        # 6. 构建并返回CSV流式响应
        return CSVBuilder.create_streaming_response(
            results=results,
            filename_prefix=file_prefix,
            column_mapping=column_mapping,
            date_columns=['timestamp'],
            exclude_columns=['id', 'updated_at']
        )

    except Exception as e:
        # 处理未预期的错误
        return CSVBuilder.create_error_response(
            message=f"服务器错误: {str(e)}",
            filename_prefix="error_arima"
        )


@router.post("/arima/import-json-results", response_model=Dict[str, Any])
async def import_json_predictions(
    background_tasks: BackgroundTasks,
    json_name: str = Query("predictions_arima.json",
                           description="JSON文件名，位于app/data/algorithm1文件夹下")
):
    """
    导入ARIMA预测结果的JSON数据

    将指定的JSON文件中的数据导入到predictions_arima_auto表

    参数:
    - **json_name**: JSON文件名，位于app/data/algorithm1文件夹下

    返回:
    - 导入任务的状态信息
    """
    # 1. 构建JSON文件的完整路径
    base_path = "app/data/algorithm1"
    json_path = os.path.join(base_path, json_name)

    # 2. 检查文件夹是否存在，不存在则创建
    if not os.path.exists(base_path):
        try:
            os.makedirs(base_path, exist_ok=True)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"无法创建数据目录: {str(e)}"
            )

    # 3. 检查JSON文件是否存在
    if not os.path.exists(json_path):
        raise HTTPException(
            status_code=404,
            detail=f"JSON文件不存在: {json_path}"
        )

    # 4. 导入JSON数据(后台任务)
    background_tasks.add_task(
        import_json_to_model,
        json_path=json_path,
        model_class=PredictionsArimaAuto,
        batch_size=100,
        encoding='utf-8'
    )

    return {
        "message": "已开始导入ARIMA预测数据",
        "status": "processing",
        "json_path": json_path
    }

# ? 可能会用到的辅助接口


@router.get("/arima/points",
            summary="获取监测点列表",
            description="获取所有可用的ARIMA预测监测点ID列表")
async def get_arima_points_list():
    """
    获取所有可用的ARIMA预测监测点ID列表

    返回：
    - **points**: 监测点ID列表
    - **count**: 监测点数量
    """
    # 查询所有不同的监测点ID
    points = await PredictionsArimaAuto.all().distinct().values_list('point_id', flat=True)
    # 对点位进行排序
    sorted_points = sorted(points)

    return {
        "points": sorted_points,
        "count": len(sorted_points)
    }


@router.get("/arima/regions",
            summary="获取区域列表",
            description="获取所有可用的ARIMA预测区域列表")
async def get_arima_regions_list():
    """
    获取所有可用的ARIMA预测区域列表

    返回：
    - **regions**: 区域列表
    - **count**: 区域数量
    """
    # 查询所有不同的区域
    regions = await PredictionsArimaAuto.all().distinct().values_list('region', flat=True)
    # 过滤None值并排序
    sorted_regions = sorted([r for r in regions if r])

    return {
        "regions": sorted_regions,
        "count": len(sorted_regions)
    }
