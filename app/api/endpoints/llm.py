from fastapi import APIRouter, Query
# from app.services.multiLLM.top_llm import main

router = APIRouter()


@router.get("/")
async def get_llm(user_questions: str = Query(..., description="用户问题")):
    """
    获取LLM的回答
    """
    # answer = main(user_questions)
    return {"answer": 1}
