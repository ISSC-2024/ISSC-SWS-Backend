import os
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Depends
from typing import Any, Dict, Optional, List, Tuple

from app.models.algorithm1.predictions_timemixer_auto import PredictionsTimeMixerAuto
from app.schemas.algorithm1.algorithm1_result import (
    PaginationInfo,
    PaginatedPredictionsTimeMixerAutos
)
from app.utils.csv_builder import CSVBuilder
from fastapi.responses import StreamingResponse

from app.utils.json_importer import import_json_to_model
from fastapi.responses import FileResponse
from datetime import datetime
from datetime import timedelta

router = APIRouter()

# 查询结果的通用函数，支持分页


# 在模块顶部添加
_MIN_TIMESTAMP = None


async def get_predictions_with_filters(
    region: Optional[str] = None,
    point_id: Optional[str] = None,
    timestep: int = 0,
    skip: int = 0,
    limit: int = 100,
    get_all: bool = False
) -> Tuple[List[PredictionsTimeMixerAuto], int]:
    """
    根据过滤条件获取预测结果，支持分页或获取全部数据

    参数:
        region: 可选的区域过滤条件
        point_id: 可选的监测点ID过滤条件
        skip: 跳过的记录数
        limit: 返回的最大记录数
        get_all: 是否返回所有符合条件的数据，为True时忽略skip和limit参数

    返回:
        Tuple[List[PredictionsTimeMixerAuto], int]: (结果列表, 总记录数)
    """
    global _MIN_TIMESTAMP
    # 构建查询条件
    query_filters = {}
    # 如果未缓存则查询数据库
    if _MIN_TIMESTAMP is None:
        min_timestamp_record = await PredictionsTimeMixerAuto.all().order_by("timestamp").first()
        _MIN_TIMESTAMP = min_timestamp_record.timestamp if min_timestamp_record else datetime.utcnow()

    # 计算时间戳（基础时间 + timestep*10秒）
    calculated_timestamp = _MIN_TIMESTAMP + timedelta(seconds=timestep*10)
    query_filters["timestamp"] = calculated_timestamp

    if region:
        query_filters["region"] = region
    if point_id:
        query_filters["point_id"] = point_id

    # 获取总记录数
    total_count = await PredictionsTimeMixerAuto.filter(**query_filters).count()

    # 根据get_all参数决定是否应用分页
    if get_all:
        results = await PredictionsTimeMixerAuto.filter(**query_filters).order_by("timestamp").all()
    else:
        results = await PredictionsTimeMixerAuto.filter(**query_filters).order_by("timestamp").offset(skip).limit(limit)

    return results, total_count


@router.get(
    "/TimeMixer",
    response_model=PaginatedPredictionsTimeMixerAutos,
    summary="获取TimeMixer预测结果",
    description="获取TimeMixer自动预测结果，支持按区域过滤和分页"
)
async def get_TimeMixer_predictions(
    region: Optional[str] = Query(None, description="区域代码，不提供则返回所有区域结果"),
    point_id: Optional[str] = Query(None, description="可选的监测点ID"),
    timestep: int = Query(0, ge=0, le=29, description="时间步长"),
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(100, ge=1, le=500, description="返回的最大记录数"),
    get_all: bool = Query(False, description="是否返回所有符合条件的数据，为true时忽略分页参数"),
):
    """
    获取TimeMixer自动预测结果，支持按区域过滤和分页

    参数:
    - **region**: 可选，区域代码，不提供则返回所有区域结果
    - **point_id**: 可选，特定监测点ID
    - **timestep**: 时间步长，默认为0，表示从起始时间戳往后的时间步，比如起始时间2023-12-02 12:12:12,时间步为1，预测的步长为10，则现在的时间为2023-12-02 12:22:12
    - **skip**: 分页偏移量，默认为0
    - **limit**: 每页记录数，默认100，最大500
    - **get_all**: 设置为true时返回所有符合条件的数据，忽略分页参数

    返回:
    - 包含分页信息和结果数据的响应
    """
    # 获取结果和总数
    results, total_count = await get_predictions_with_filters(
        region, point_id, timestep, skip, limit, get_all
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
        return PaginatedPredictionsTimeMixerAutos(
            pagination=pagination,
            data=[]
        )

    return PaginatedPredictionsTimeMixerAutos(
        pagination=pagination,
        data=results
    )


@router.get("/TimeMixer/results/download-csv", response_class=StreamingResponse)
async def download_TimeMixer_predictions_csv(
    region: Optional[str] = Query(None, description="区域过滤"),
    point_id: Optional[str] = Query(None, description="监测点ID"),
    filename: Optional[str] = Query(None, description="自定义文件名前缀"),
    timestep: int = Query(0, ge=0, le=29, description="时间步长"),
    localize: bool = Query(False, description="是否使用中文列名"),
):
    """
    下载TimeMixer预测结果为CSV格式

    参数:
        region: 可选的区域过滤
        point_id: 可选的监测点ID
        filename: 自定义文件名前缀
        timestep: 时间步长
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

        global _MIN_TIMESTAMP

        # 如果未缓存则查询数据库
        if _MIN_TIMESTAMP is None:
            min_timestamp_record = await PredictionsTimeMixerAuto.all().order_by("timestamp").first()
            _MIN_TIMESTAMP = min_timestamp_record.timestamp if min_timestamp_record else datetime.utcnow()

        # 计算时间戳（基础时间 + timestep*10秒）
        calculated_timestamp = _MIN_TIMESTAMP + timedelta(seconds=timestep*10)
        query_filters["timestamp"] = calculated_timestamp

        if region:
            query_filters["region"] = region
            filter_desc.append(f"区域={region}")

        if point_id:
            query_filters["point_id"] = point_id
            filter_desc.append(f"监测点={point_id}")

        # 2. 查询所有符合条件的数据
        results = await PredictionsTimeMixerAuto.filter(**query_filters).order_by("timestamp").all()

        # 3. 判断是否有数据
        if not results:
            return CSVBuilder.create_error_response(
                message=f"未找到符合条件的数据: {', '.join(filter_desc) if filter_desc else '所有记录'}",
                filename_prefix="no_data"
            )

        # 4. 生成文件名前缀
        if not filename:
            parts = ["TimeMixer"]

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
            filename_prefix="error_TimeMixer"
        )


@router.post("/TimeMixer/import-json-results", response_model=Dict[str, Any])
async def import_json_predictions(
    background_tasks: BackgroundTasks,
    json_name: str = Query("predictions_TimeMixer_auto.json",
                           description="JSON文件名，位于app/data/algorithm1文件夹下")
):
    """
    导入TimeMixer预测结果的JSON数据

    将指定的JSON文件中的数据导入到predictions_TimeMixer_auto表

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
        model_class=PredictionsTimeMixerAuto,
        batch_size=100,
        encoding='utf-8'
    )

    return {
        "message": "已开始导入TimeMixer预测数据",
        "status": "processing",
        "json_path": json_path
    }

# ? 可能会用到的辅助接口


@router.get("/TimeMixer/points",
            summary="获取监测点列表",
            description="获取所有可用的TimeMixer预测监测点ID列表")
async def get_TimeMixer_points_list():
    """
    获取所有可用的TimeMixer预测监测点ID列表

    返回：
    - **points**: 监测点ID列表
    - **count**: 监测点数量
    """
    # 查询所有不同的监测点ID
    points = await PredictionsTimeMixerAuto.all().distinct().values_list('point_id', flat=True)
    # 对点位进行排序
    sorted_points = sorted(points)

    return {
        "points": sorted_points,
        "count": len(sorted_points)
    }


@router.get("/TimeMixer/regions",
            summary="获取区域列表",
            description="获取所有可用的TimeMixer预测区域列表")
async def get_TimeMixer_regions_list():
    """
    获取所有可用的TimeMixer预测区域列表

    返回：
    - **regions**: 区域列表
    - **count**: 区域数量
    """
    # 查询所有不同的区域
    regions = await PredictionsTimeMixerAuto.all().distinct().values_list('region', flat=True)
    # 过滤None值并排序
    sorted_regions = sorted([r for r in regions if r])

    return {
        "regions": sorted_regions,
        "count": len(sorted_regions)
    }


@router.get("/TimeMixer/prediction-chart")
async def get_prediction_chart(
    point_id: str = Query(..., description="监测点ID"),
    timestamp: str = Query(..., description="时间戳，格式：YYYY-MM-DD HH:mm:ss")
):
    """获取预测图表"""
    try:
        # 构建图片路径
        base_path = "app/data/algorithm1/TimeMixer"
        # Format timestamp as part of filename (e.g., RMS001_2025-03-02.png)
        image_name = "sensor_data.png"
        image_path = os.path.join(base_path, point_id, image_name)

        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=404,
                detail=f"图表不存在: {image_name}"
            )

        return FileResponse(
            image_path,
            media_type="image/png",
            filename=image_name
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取图表失败: {str(e)}"
        )
