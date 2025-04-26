from tortoise.models import Model
from tortoise import fields


class Algorithm2Config(Model):
    # 主键，自动递增
    config_id = fields.IntField(pk=True)

    tree_count = fields.IntField(description="决策树数量")
    max_depth = fields.IntField(description="树最大深度")
    sensitivity = fields.FloatField(description="偏离敏感度")

    updated_at = fields.DatetimeField(auto_now=True, description="最后更新时间")

    class Meta:
        table = "algorithm2_config"
        # 唯一约束，一种参数组合是唯一的(自动创建索引)
        unique_together = [("tree_count", "max_depth", "sensitivity")]
