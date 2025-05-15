# /app/models/__init__.py

# 该文件用于初始化模型模块。

# 导入所有模型以确保 Aerich 能够检测到它们
from app.models.algorithm1.predictions_timemixer_auto import PredictionsTimeMixerAuto
from app.models.algorithm2.algorithm2_config import Algorithm2Config
from app.models.algorithm2.algorithm2_result import Algorithm2Result
from app.models.algorithm3.algorithm3_config import Algorithm3Config
from app.models.algorithm3.algorithm3_result import Algorithm3Result
from app.models.llm.conversation import Conversation
from app.models.llm.message import Message
