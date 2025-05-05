from tortoise.models import Model
from tortoise import fields
from datetime import datetime


class Conversation(Model):
    """对话历史记录模型"""
    conversation_id = fields.IntField(pk=True)
    title = fields.CharField(max_length=255, description="对话标题", null=False)
    model = fields.CharField(max_length=50, description="使用的AI模型", null=False)
    created_at = fields.DatetimeField(auto_now_add=True, description="创建时间")
    updated_at = fields.DatetimeField(auto_now=True, description="更新时间")

    # 定义与Message模型的关系
    messages = fields.ReverseRelation["Message"]

    class Meta:
        table = "conversations"
        description = "AI对话历史记录"
