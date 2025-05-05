# 请求和响应模型
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from app.utils.timezone import to_cn_timezone


class ConversationCreate(BaseModel):
    title: str = Field(..., description="对话标题")
    model: str = Field(..., description="使用的AI模型")


class ConversationResponse(BaseModel):
    id: int = Field(..., description="对话ID")
    title: str = Field(..., description="对话标题")
    model: str = Field(..., description="使用的AI模型")
    createdAt: datetime = Field(..., description="创建时间")
    updatedAt: datetime = Field(..., description="更新时间")
    messageCount: Optional[int] = Field(None, description="消息数量")

    class Config:
        json_encoders = {
            # 确保datetime以东八区时间返回
            datetime: lambda dt: to_cn_timezone(dt).isoformat()
        }


class ConversationUpdate(BaseModel):
    title: str = Field(..., description="新的对话标题")
