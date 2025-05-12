
from datetime import datetime
from pydantic import BaseModel


class PredictionsArimaAutoBase(BaseModel):
    timestamp: datetime
    point_id: str
    region: str

    temperature: float
    pressure: float
    flow_rate: float
    level: float

    gas_type: str
    gas_concentration: float


class PredictionsArimaAutoResponse(PredictionsArimaAutoBase):

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
class PaginatedPredictionsArimaAutos(BaseModel):
    """带分页信息的结果列表响应模型"""
    pagination: PaginationInfo
    data: list[PredictionsArimaAutoResponse]
