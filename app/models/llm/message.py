from tortoise.models import Model
from tortoise import fields
from tortoise.indexes import Index
from enum import Enum


class MessageRole(str, Enum):
    """消息角色枚举"""
    USER = "user"
    ASSISTANT = "assistant"


class Message(Model):
    """对话消息记录模型"""
    message_id = fields.IntField(pk=True)
    conversation = fields.ForeignKeyField(
        'models.Conversation',
        related_name='messages',
        description="所属对话",
        on_delete=fields.CASCADE,
        index=True,
        field_name="conversation_id"
    )
    role = fields.CharEnumField(
        MessageRole,
        description="消息角色",
        max_length=10,
        null=False
    )
    content = fields.TextField(description="消息内容", null=False)
    thinking = fields.TextField(description="AI思考过程", null=True)
    is_thinking_expanded = fields.BooleanField(
        default=False, description="思考过程是否展开", null=True)
    model = fields.CharField(max_length=50, description="使用的AI模型", null=True)
    timestamp = fields.DatetimeField(auto_now_add=True, description="消息时间")

    class Meta:
        table = "messages"
        description = "对话消息记录"
        indexes = [
            # 创建联合索引
            Index(fields=["conversation_id", "timestamp"],
                  name="idx_conversation_timestamp")
        ]
