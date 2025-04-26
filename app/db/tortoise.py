from tortoise import Tortoise
from app.core.config import config

# Tortoise ORM配置供Aerich使用
TORTOISE_ORM = {
    "connections": {"default": config.DATABASE_URL},
    "apps": {
        "models": {
            "models": ["app.models", "aerich.models"],
            "default_connection": "default",
        }
    }
}

async def init_db():
    """
    初始化数据库连接
    """
    await Tortoise.init(
        db_url=config.DATABASE_URL,
        modules={"models": ["app.models", "aerich.models"]}
    )
    await Tortoise.generate_schemas()

async def close_db():
    """
    关闭数据库连接
    """
    await Tortoise.close_connections()