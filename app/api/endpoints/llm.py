from fastapi import APIRouter, Query, Path
from app.utils.httpx_client import HttpxClient
from typing import List, Dict, Any
from app.core.config import Config

router = APIRouter()

# LLM APP所在
async_client = HttpxClient(base_url=Config.LLM_APP_URL, timeout=90)

# 支持的LLM模型列表
LLM_MODELS = [
    {"endpoint": "top-llm", "name": "主LLM", "description": "获取主LLM的回答"},
    {"endpoint": "sub-llm1", "name": "子LLM1", "description": "获取子LLM1的回答"},
    {"endpoint": "sub-llm2", "name": "子LLM2", "description": "获取子LLM2的回答"},
    {"endpoint": "sub-llm3", "name": "子LLM3", "description": "获取子LLM3的回答"},
    {"endpoint": "sub-llm4", "name": "子LLM4", "description": "获取子LLM4的回答"},
    {"endpoint": "sub-llm5", "name": "子LLM5", "description": "获取子LLM5的回答"},
]


async def query_llm(endpoint: str, user_question: str) -> Dict[str, Any]:
    """
    通用LLM查询函数

    Args:
        endpoint: LLM接口地址
        user_question: 用户问题

    Returns:
        LLM的响应
    """
    resp = await async_client.async_get(
        f'/{endpoint}',
        params={"user_question": user_question}
    )
    return resp.json()


@router.get("/models")
async def get_available_models() -> List[Dict[str, str]]:
    """获取所有可用的LLM模型信息"""
    return [
        {"endpoint": model["endpoint"], "name": model["name"]}
        for model in LLM_MODELS
    ]

# 通用LLM接口 - 可以通过路径参数选择LLM模型


@router.get("/query/{model_endpoint}")
async def query_model(
    model_endpoint: str = Path(..., description="LLM模型端点名称"),
    user_question: str = Query(..., description="用户问题")
):
    """
    通用LLM查询接口，可以通过路径参数指定要使用的LLM模型

    例如: /api/llm/query/top-llm?user_question=你好
    """
    # 验证模型是否存在
    valid_endpoints = [model["endpoint"] for model in LLM_MODELS]
    if model_endpoint not in valid_endpoints:
        return {"error": f"模型 '{model_endpoint}' 不存在", "valid_models": valid_endpoints}

    return await query_llm(model_endpoint, user_question)
