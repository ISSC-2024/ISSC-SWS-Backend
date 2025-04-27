from tortoise.models import Model
from tortoise import fields
from tortoise.indexes import Index


class Algorithm2Result(Model):
    # 主键，自动递增
    result_id = fields.BigIntField(pk=True)
    # 外键，指向 Algorithm2Config 表，建立索引
    config = fields.ForeignKeyField(
        'models.Algorithm2Config', related_name='results', index=True, field_name='config_id')

    point_id = fields.CharField(max_length=10, description="监测点（传感器）ID")
    area_code = fields.CharField(max_length=5, description="区域编码")
    pred_risk = fields.CharField(max_length=10, description="预测风险等级")
    weight = fields.FloatField(description="权重")

    # 最后更新时间
    updated_at = fields.DatetimeField(auto_now=True, description="最后更新时间")

    class Meta:
        table = "algorithm2_result"
        # config_id和area_code联合索引
        indexs = [
            Index(fields=["config_id", "area_code"]),
        ]
