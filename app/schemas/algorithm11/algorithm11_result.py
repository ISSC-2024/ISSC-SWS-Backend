
from datetime import datetime
from pydantic import BaseModel


class PredictionsTimeMixerAutoBase(BaseModel):
    timestamp: datetime
    region: str
    sensor_type: str
    measurement: str
    value: float


class PredictionsTimeMixerAutoResponse(PredictionsTimeMixerAutoBase):

    class Config:
        from_attributes = True


# 分页模型
class PaginationInfo(BaseModel):
    """分页信息模型"""
    total: int  # 总记录数
    skip: int  # 当前偏移量
    limit: int  # 页大小
    has_more: bool  # 是否还有更多数据


# 带分页的响应模型
class PaginatedPredictionsTimeMixerAutoIrons(BaseModel):
    """带分页信息的结果列表响应模型"""
    pagination: PaginationInfo
    data: list[PredictionsTimeMixerAutoResponse]
