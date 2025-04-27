from tortoise.models import Model
from tortoise import fields


class Algorithm2Config(Model):
    # 主键，自动递增
    config_id = fields.IntField(pk=True)

    # 非空
    tree_count = fields.IntField(description="决策树数量", null=False)
    max_depth = fields.IntField(description="树最大深度", null=False)
    sensitivity = fields.FloatField(description="偏离敏感度", null=False)

    updated_at = fields.DatetimeField(auto_now=True, description="最后更新时间")

    class Meta:
        table = "algorithm2_config"
        # 唯一约束，一种参数组合是唯一的(自动创建索引)
        unique_together = [("tree_count", "max_depth", "sensitivity")]
