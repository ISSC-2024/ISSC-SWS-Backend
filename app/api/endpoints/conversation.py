from fastapi import APIRouter, HTTPException, Depends
from typing import List
from datetime import datetime
from app.models.llm.conversation import Conversation
from app.models.llm.message import Message
from tortoise.functions import Count

from app.schemas.llm.conversation import ConversationCreate, ConversationResponse, ConversationUpdate
from app.schemas.llm.message import MessageCreate, MessageResponse
from app.utils.timezone import CN_TIMEZONE

router = APIRouter()


async def get_conversation_or_404(conversation_id: int) -> Conversation:
    try:
        return await Conversation.get(conversation_id=conversation_id)
    except:
        raise HTTPException(status_code=404, detail="对话不存在")

# 创建新对话


@router.post("", response_model=ConversationResponse, status_code=201)
async def create_conversation(conversation: ConversationCreate):
    """创建新对话"""
    new_conversation = await Conversation.create(title=conversation.title, model=conversation.model)
    return {
        "id": new_conversation.conversation_id,
        "title": new_conversation.title,
        "model": conversation.model,
        "createdAt": new_conversation.created_at,
        "updatedAt": new_conversation.updated_at,
        "messageCount": 0
    }

# 获取所有对话列表


@router.get("", response_model=List[ConversationResponse])
async def get_conversations():
    """获取所有对话列表，包含每个对话的消息数量"""
    # 使用annotate计算消息数
    conversations = await Conversation.all().annotate(
        message_count=Count('messages')
    ).order_by('-updated_at')

    return [
        {
            "id": conv.conversation_id,
            "title": conv.title,
            "model": conv.model,
            "createdAt": conv.created_at,
            "updatedAt": conv.updated_at,
            "messageCount": conv.message_count
        } for conv in conversations
    ]

# 重命名对话


@router.put("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_update: ConversationUpdate,
    conversation: Conversation = Depends(get_conversation_or_404)
):
    """重命名对话"""
    conversation.title = conversation_update.title
    await conversation.save()

    # 获取消息数量
    message_count = await Message.filter(conversation_id=conversation.conversation_id).count()

    return {
        "id": conversation.conversation_id,
        "title": conversation.title,
        "model": conversation.model,
        "createdAt": conversation.created_at,
        "updatedAt": conversation.updated_at,
        "messageCount": message_count
    }

# 删除对话


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation: Conversation = Depends(get_conversation_or_404)):
    """删除对话"""
    await conversation.delete()
    return None

# 获取特定对话的所有消息


@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(conversation: Conversation = Depends(get_conversation_or_404)):
    """获取特定对话的所有消息"""
    messages = await Message.filter(conversation_id=conversation.conversation_id).order_by('timestamp')

    return [
        {
            "id": msg.message_id,
            "role": msg.role,
            "content": msg.content,
            "thinking": msg.thinking,
            "isThinkingExpanded": msg.is_thinking_expanded,
            "model": msg.model,
            "timestamp": msg.timestamp,
            "conversation_id": msg.conversation_id
        } for msg in messages
    ]

# 添加新消息


@router.post("/{conversation_id}/messages", response_model=MessageResponse, status_code=201)
async def create_message(
    message: MessageCreate,
    conversation: Conversation = Depends(get_conversation_or_404)
):
    """添加新消息到指定对话"""
    # 为了确保updated_at字段正确更新，需要手动设置
    now = datetime.now(CN_TIMEZONE)

    new_message = await Message.create(
        conversation=conversation,
        role=message.role,
        content=message.content,
        thinking=message.thinking,
        model=message.model
    )

    # 更新对话的更新时间
    conversation.updated_at = now
    await conversation.save()

    return {
        "id": new_message.message_id,
        "role": new_message.role,
        "content": new_message.content,
        "thinking": new_message.thinking,
        "isThinkingExpanded": new_message.is_thinking_expanded,
        "model": new_message.model,
        "timestamp": new_message.timestamp,
        "conversation_id": new_message.conversation_id
    }
