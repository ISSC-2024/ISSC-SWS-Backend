from tortoise import fields
from tortoise.models import Model


class Algorithm1Config(Model):
    """
    算法1配置表模型

    包含必填参数:
    - model_id: 传感器编号
    - task_name: 任务类型

    可选参数:
    - seq_len: 输入序列长度
    - pred_len: 预测序列长度
    - train_epochs: 训练的epoch数
    - batch_size: 批数量
    """
    # 主键
    config_id = fields.IntField(pk=True, description="配置ID")

    # 必填参数
    algorithm = fields.CharField(
        max_length=20,
        description="算法名称",
        choices=[
            'TimeMixer',
            'TimesNet',
        ],
        null=False
    )
    model_id = fields.CharField(max_length=10, description="传感器编号")
    task_name = fields.CharField(max_length=50, description="任务类型")

    # 可选参数
    seq_len = fields.IntField(null=True, description="输入序列长度")
    pred_len = fields.IntField(null=True, description="预测序列长度")
    train_epochs = fields.IntField(null=True, description="训练的epoch数")
    batch_size = fields.IntField(null=True, description="批数量")

    # 记录创建和更新时间
    updated_at = fields.DatetimeField(auto_now=True, description="最后更新时间")

    class Meta:
        table = "algorithm1_config"
        description = "算法1配置表"
