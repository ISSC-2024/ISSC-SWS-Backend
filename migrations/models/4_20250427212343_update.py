from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS `algorithm3_config` (
    `config_id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `algorithm` VARCHAR(20) NOT NULL COMMENT '算法名称',
    `learning_rate` DOUBLE NOT NULL COMMENT '学习率',
    `max_depth` INT COMMENT '最大深度',
    `max_epochs` INT COMMENT '最大迭代次数',
    `updated_at` DATETIME(6) NOT NULL COMMENT '最后更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    UNIQUE KEY `uid_algorithm3__algorit_6c9108` (`algorithm`, `learning_rate`, `max_depth`, `max_epochs`)
) CHARACTER SET utf8mb4;
        CREATE TABLE IF NOT EXISTS `algorithm3_result` (
    `result_id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `timestamp` DATETIME(6) NOT NULL COMMENT '时间戳',
    `point_id` VARCHAR(10) NOT NULL COMMENT '监测点（传感器）ID',
    `temperature` DOUBLE NOT NULL COMMENT '温度',
    `pressure` DOUBLE NOT NULL COMMENT '压力',
    `flow_rate` DOUBLE NOT NULL COMMENT '流速',
    `level` DOUBLE NOT NULL COMMENT '液位',
    `gas_type` VARCHAR(10) NOT NULL COMMENT '气体类型',
    `gas_concentration` DOUBLE NOT NULL COMMENT '气体浓度',
    `risk_level` VARCHAR(15) NOT NULL COMMENT '风险等级',
    `risk_level_name` VARCHAR(10) NOT NULL COMMENT '风险等级名称',
    `message` LONGTEXT NOT NULL COMMENT '风险描述信息',
    `updated_at` DATETIME(6) NOT NULL COMMENT '最后更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    `config_id` INT NOT NULL,
    CONSTRAINT `fk_algorith_algorith_2885caad` FOREIGN KEY (`config_id`) REFERENCES `algorithm3_config` (`config_id`) ON DELETE CASCADE,
    KEY `idx_algorithm3__config__2b3099` (`config_id`),
    KEY `idx_algorithm3_result_config_point` (`point_id`, `config_id`),
    KEY `idx_algorithm3_result_risk` (`risk_level`)
) CHARACTER SET utf8mb4;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS `algorithm3_result`;
        DROP TABLE IF EXISTS `algorithm3_config`;"""
