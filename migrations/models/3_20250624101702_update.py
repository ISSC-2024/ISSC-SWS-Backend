from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS `predictions_timemixer_auto_iron` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '记录ID',
    `timestamp` DATETIME(6) NOT NULL COMMENT '预测时间点',
    `region` VARCHAR(10) COMMENT '区域',
    `sensor_type` VARCHAR(10) COMMENT '传感器类型',
    `measurement` VARCHAR(10) COMMENT '测量量',
    `value` DOUBLE COMMENT '测量值',
    `updated_at` DATETIME(6) NOT NULL COMMENT '最后更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    KEY `idx_predictions_timesta_48ed01` (`timestamp`),
    KEY `idx_predictions_measure_1f6c92` (`measurement`),
    KEY `idx_predictions_region_98120a` (`region`)
) CHARACTER SET utf8mb4 COMMENT='TimeMixer自动预测结果表模型';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS `predictions_timemixer_auto_iron`;"""
