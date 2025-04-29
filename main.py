from fastapi import FastAPI, Request
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

app = FastAPI(lifespan=lifespan, title="全域互联的工业智能体协同平台后端")

# 配置CORS - 只允许同IP访问


@app.middleware("http")
async def add_cors_header(request: Request, call_next):
    # 获取请求的主机IP
    host = request.headers.get("host", "").split(":")[0]
    print(f"Host: {host}")
    # 构建允许的origin
    origins = [f"http://{host}", f"https://{host}",
               f"http://{host}:*", f"https://{host}:*"]

    response = await call_next(request)

    # 获取请求的Origin
    origin = request.headers.get("origin", "")
    print(f"Origin: {origin}")
    # 检查Origin是否来自同一IP
    for allowed_origin in origins:
        if allowed_origin.replace("*", "") in origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            break

    return response

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

# 直接运行此脚本，启动服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', host="localhost", port=8000, reload=True)
