from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.tortoise import init_db, close_db
from app.api.endpoints import algorithm1, algorithm2, algorithm3, algorithm4, llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化数据库
    await init_db()
    yield
    # 关闭时关闭数据库连接
    await close_db()

app = FastAPI(lifespan=lifespan)

# 注册路由
app.include_router(algorithm1.router,
                   prefix="/api/algorithm1", tags=["Algorithm1"])
app.include_router(algorithm2.router,
                   prefix="/api/algorithm2", tags=["Algorithm2"])
app.include_router(algorithm3.router,
                   prefix="/api/algorithm3", tags=["Algorithm3"])
app.include_router(algorithm4.router,
                   prefix="/api/algorithm4", tags=["Algorithm4"])
app.include_router(llm.router, prefix="/api/llm", tags=["LLM"])
