from tortoise.models import Model
from tortoise import fields
from tortoise.validators import Validator
from tortoise.exceptions import ValidationError
from typing import Optional, Any


class Algorithm3Validator(Validator):
    """
    验证不同算法类型的字段约束:
    - xgboost/lightGBM: max_depth必填，max_epochs可为空
    - TabNet: max_depth可为空，max_epochs必填
    """

    def __call__(self, value: Any) -> None:
        # 获取当前模型实例
        instance = self.model
        algorithm = getattr(instance, 'algorithm', None)
        max_depth = getattr(instance, 'max_depth', None)
        max_epochs = getattr(instance, 'max_epochs', None)

        # xgboost和lightGBM算法的验证
        if algorithm in ['xgboost', 'lightGBM']:
            if max_depth is None:
                raise ValidationError(
                    f"{algorithm}算法需要设置max_depth参数")

        # TabNet算法的验证
        elif algorithm == 'TabNet':
            if max_epochs is None:
                raise ValidationError(
                    f"{algorithm}算法需要设置max_epochs参数")


class Algorithm3Config(Model):
    # 主键，自动递增
    config_id = fields.IntField(pk=True)

    # 算法名称，只能为xgboost、lightGBM或TabNet
    algorithm = fields.CharField(
        max_length=20,
        description="算法名称",
        choices=[
            'xgboost',
            'lightGBM',
            'TabNet'
        ],
        null=False
    )

    # 学习率，所有算法都需要
    learning_rate = fields.FloatField(description="学习率", null=False)

    # 最大深度，只有xgboost和lightGBM算法需要
    max_depth = fields.IntField(
        description="最大深度",
        null=True
    )

    # 最大迭代次数，只有TabNet算法需要
    max_epochs = fields.IntField(
        description="最大迭代次数",
        null=True
    )

    # 最后更新时间
    updated_at = fields.DatetimeField(auto_now=True, description="最后更新时间")

    class Meta:
        table = "algorithm3_config"

        # 验证器列表
        validators = [Algorithm3Validator()]

        # 唯一约束
        unique_together = [("algorithm", "learning_rate",
                            "max_depth", "max_epochs")]

    async def save(self, *args, **kwargs) -> None:
        # 保存前强制设置某些字段为None，确保数据一致性
        if self.algorithm in ['xgboost', 'lightGBM']:
            self.max_epochs = None
        elif self.algorithm == 'TabNet':
            self.max_depth = None

        await super().save(*args, **kwargs)
