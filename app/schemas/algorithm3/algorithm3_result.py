from pydantic import BaseModel
from datetime import datetime
from typing import List, Generic, TypeVar, Dict, Any

# 保留现有模型


class Algorithm3ResultBase(BaseModel):
    timestamp: datetime
    region: str
    risk_level: str
    message: str | None


class Algorithm3ResultResponse(Algorithm3ResultBase):
    config_id: int


# 添加新的分页模型
class PaginationInfo(BaseModel):
    """分页信息模型"""
    total: int  # 总记录数
    skip: int  # 当前偏移量
    limit: int  # 页大小
    has_more: bool  # 是否还有更多数据


# 添加带分页的响应模型
class PaginatedAlgorithm3Results(BaseModel):
    """带分页信息的结果列表响应模型"""
    pagination: PaginationInfo
    data: List[Algorithm3ResultResponse]


# 通用分页响应模型 (可选，如果需要在多个地方复用)
T = TypeVar('T')


class PaginatedResponse(BaseModel, Generic[T]):
    """通用的分页响应模型，可用于各种数据类型"""
    pagination: PaginationInfo
    data: List[T]
