from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS `algorithm2_config` (
    `config_id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `tree_count` INT NOT NULL COMMENT '决策树数量',
    `max_depth` INT NOT NULL COMMENT '树最大深度',
    `sensitivity` DOUBLE NOT NULL COMMENT '偏离敏感度',
    `updated_at` DATETIME(6) NOT NULL COMMENT '最后更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    UNIQUE KEY `uid_algorithm2__tree_co_68b6eb` (`tree_count`, `max_depth`, `sensitivity`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `algorithm2_result` (
    `result_id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `point_id` VARCHAR(10) NOT NULL COMMENT '监测点（传感器）ID',
    `area_code` VARCHAR(5) NOT NULL COMMENT '区域编码',
    `pred_risk` VARCHAR(10) NOT NULL COMMENT '预测风险等级',
    `weight` DOUBLE NOT NULL COMMENT '权重',
    `updated_at` DATETIME(6) NOT NULL COMMENT '最后更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    `config_id_id` INT NOT NULL,
    CONSTRAINT `fk_algorith_algorith_fc7bce90` FOREIGN KEY (`config_id_id`) REFERENCES `algorithm2_config` (`config_id`) ON DELETE CASCADE,
    KEY `idx_algorithm2__config__cf3b9e` (`config_id_id`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `aerich` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `version` VARCHAR(255) NOT NULL,
    `app` VARCHAR(100) NOT NULL,
    `content` JSON NOT NULL
) CHARACTER SET utf8mb4;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """
