from tortoise import fields
from tortoise.models import Model
from typing import Optional, Dict, Any


class PredictionsArimaAuto(Model):
    """
    ARIMA自动预测结果表模型

    存储ARIMA模型自动生成的预测数据，包括时间戳、点位ID和各种传感器数据
    """
    id = fields.IntField(pk=True, description="记录ID")

    # 时间和位置标识
    timestamp = fields.DatetimeField(description="预测时间点")
    point_id = fields.CharField(max_length=10, description="监测点ID")

    # 传感器数据
    temperature = fields.FloatField(null=True, description="温度")
    pressure = fields.FloatField(null=True, description="压力")
    flow_rate = fields.FloatField(null=True, description="流速")
    level = fields.FloatField(null=True, description="液位")

    # 气体数据
    gas_type = fields.CharField(max_length=10, null=True, description="气体类型")
    gas_concentration = fields.FloatField(null=True, description="气体浓度")

    # 元数据
    updated_at = fields.DatetimeField(auto_now=True, description="最后更新时间")

    class Meta:
        table = "predictions_arima_auto"
        description = "ARIMA自动预测结果表"
        indexes = [
            ("timestamp", "point_id"),
            "timestamp",
            "point_id"
        ]
