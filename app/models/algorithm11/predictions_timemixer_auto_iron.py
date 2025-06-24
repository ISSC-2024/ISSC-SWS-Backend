from tortoise import fields
from tortoise.models import Model
from typing import Any
from tortoise.indexes import Index


class PredictionsTimeMixerAutoIron(Model):
    """
    TimeMixer自动预测结果表模型

    存储TimeMixer模型自动生成的预测数据，包括时间戳、点位ID和各种传感器数据
    """
    id = fields.IntField(pk=True, description="记录ID")

    # 时间和位置标识
    timestamp = fields.DatetimeField(description="预测时间点")
    region = fields.CharField(max_length=10, description="区域", null=True)
    sensor_type = fields.CharField(max_length=10, description="传感器类型", null=True)
    measurement = fields.CharField(max_length=10, description="测量量",null=True)
    value = fields.FloatField(null=True, description="测量值")

    # 元数据
    updated_at = fields.DatetimeField(auto_now=True, description="最后更新时间")

    class Meta:
        table = "predictions_timemixer_auto_iron"
        description = "TimeMixer自动预测结果表"
        indexes = [
            Index(fields=["timestamp"]),
            Index(fields=["measurement"]),
            Index(fields=["region"]),
        ]

    async def save(self, *args: Any, **kwargs: Any) -> None:
        await super().save(*args, **kwargs)
