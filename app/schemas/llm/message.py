from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from app.models.llm.message import MessageRole
from app.utils.timezone import to_cn_timezone


class MessageCreate(BaseModel):
    role: MessageRole = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")
    thinking: Optional[str] = Field(None, description="思考过程")
    model: Optional[str] = Field(None, description="使用的AI模型")


class MessageResponse(BaseModel):
    id: int = Field(..., description="消息ID")
    role: MessageRole = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")
    thinking: Optional[str] = Field(None, description="思考过程")
    isThinkingExpanded: Optional[bool] = Field(None, description="思考过程是否展开")
    model: Optional[str] = Field(None, description="使用的AI模型")
    timestamp: datetime = Field(..., description="消息时间")
    conversation_id: int = Field(..., description="所属对话ID")

    class Config:
        json_encoders = {
            # 确保datetime以东八区时间返回
            datetime: lambda dt: to_cn_timezone(dt).isoformat()
        }
