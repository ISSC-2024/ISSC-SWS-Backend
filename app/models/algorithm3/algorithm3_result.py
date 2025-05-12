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
    region = fields.CharField(max_length=10, description="区域", null=False)
    risk_level = fields.CharField(
        max_length=15, description="风险等级", null=False)
    message = fields.TextField(description="风险描述信息", null=True)

    # 最后更新时间
    updated_at = fields.DatetimeField(auto_now=True, description="最后更新时间")

    class Meta:
        table = "algorithm3_result"
        # 索引
        indexes = [
            # config_id和region联合索引
            Index(fields=["config_id", "region"],
                  name="idx_algorithm3_result_config_region"),
            # risk_level索引
            Index(fields=["risk_level"], name="idx_algorithm3_result_risk"),
        ]
