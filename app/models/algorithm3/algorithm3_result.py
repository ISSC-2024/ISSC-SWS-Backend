from tortoise.models import Model
from tortoise import fields
from tortoise.indexes import Index
from datetime import datetime


class Algorithm3Result(Model):
    # 主键，自动递增
    result_id = fields.BigIntField(pk=True)

    # 外键，指向 Algorithm3Config 表，建立索引
    config = fields.ForeignKeyField(
        'models.Algorithm3Config', related_name='results', index=True, field_name='config_id')

    timestamp = fields.DatetimeField(description="时间戳", null=False)
    point_id = fields.CharField(
        max_length=10, description="监测点（传感器）ID", null=False)
    temperature = fields.FloatField(description="温度", null=False)
    pressure = fields.FloatField(description="压力", null=False)
    flow_rate = fields.FloatField(description="流速", null=False)
    level = fields.FloatField(description="液位", null=False)
    gas_type = fields.CharField(max_length=10, description="气体类型", null=False)
    gas_concentration = fields.FloatField(description="气体浓度", null=False)
    risk_level = fields.CharField(
        max_length=15, description="风险等级", null=False)
    risk_level_name = fields.CharField(
        max_length=10, description="风险等级名称", null=False)
    message = fields.TextField(description="风险描述信息", null=False)

    # 最后更新时间
    updated_at = fields.DatetimeField(auto_now=True, description="最后更新时间")

    class Meta:
        table = "algorithm3_result"
        # 索引
        indexes = [
            # config_id和point_id联合索引
            Index(fields={"config_id", "point_id"},
                  name="idx_algorithm3_result_config_point"),
            # risk_level索引
            Index(fields={"risk_level"}, name="idx_algorithm3_result_risk"),
        ]
