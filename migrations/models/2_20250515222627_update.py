from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS `predictions_timemixer_auto` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '记录ID',
    `timestamp` DATETIME(6) NOT NULL COMMENT '预测时间点',
    `point_id` VARCHAR(10) NOT NULL COMMENT '监测点ID',
    `region` VARCHAR(10) COMMENT '区域',
    `temperature` DOUBLE COMMENT '温度',
    `pressure` DOUBLE COMMENT '压力',
    `flow_rate` DOUBLE COMMENT '流速',
    `level` DOUBLE COMMENT '液位',
    `gas_type` VARCHAR(10) COMMENT '气体类型',
    `gas_concentration` DOUBLE COMMENT '气体浓度',
    `updated_at` DATETIME(6) NOT NULL COMMENT '最后更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    KEY `idx_predictions_timesta_53e64d` (`timestamp`),
    KEY `idx_predictions_point_i_38482f` (`point_id`),
    KEY `idx_predictions_region_05cc77` (`region`)
) CHARACTER SET utf8mb4 COMMENT='TimeMixer自动预测结果表模型';
        DROP TABLE IF EXISTS `predictions_arima_auto`;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS `predictions_timemixer_auto`;"""
