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
    `config_id` INT NOT NULL,
    CONSTRAINT `fk_algorith_algorith_8f4a903a` FOREIGN KEY (`config_id`) REFERENCES `algorithm2_config` (`config_id`) ON DELETE CASCADE,
    KEY `idx_algorithm2__config__1f0502` (`config_id`)
) CHARACTER SET utf8mb4;
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
    `region` VARCHAR(10) NOT NULL COMMENT '区域',
    `risk_level` VARCHAR(15) NOT NULL COMMENT '风险等级',
    `message` LONGTEXT COMMENT '风险描述信息',
    `updated_at` DATETIME(6) NOT NULL COMMENT '最后更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    `config_id` INT NOT NULL,
    CONSTRAINT `fk_algorith_algorith_2885caad` FOREIGN KEY (`config_id`) REFERENCES `algorithm3_config` (`config_id`) ON DELETE CASCADE,
    KEY `idx_algorithm3__config__2b3099` (`config_id`),
    KEY `idx_algorithm3_result_config_region` (`config_id`, `region`),
    KEY `idx_algorithm3_result_risk` (`risk_level`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `conversations` (
    `conversation_id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `title` VARCHAR(255) NOT NULL COMMENT '对话标题',
    `model` VARCHAR(50) NOT NULL COMMENT '使用的AI模型',
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)
) CHARACTER SET utf8mb4 COMMENT='对话历史记录模型';
CREATE TABLE IF NOT EXISTS `messages` (
    `message_id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `role` VARCHAR(10) NOT NULL COMMENT '消息角色',
    `content` LONGTEXT NOT NULL COMMENT '消息内容',
    `thinking` LONGTEXT COMMENT 'AI思考过程',
    `is_thinking_expanded` BOOL COMMENT '思考过程是否展开' DEFAULT 0,
    `model` VARCHAR(50) COMMENT '使用的AI模型',
    `timestamp` DATETIME(6) NOT NULL COMMENT '消息时间' DEFAULT CURRENT_TIMESTAMP(6),
    `conversation_id` INT NOT NULL COMMENT '所属对话',
    CONSTRAINT `fk_messages_conversa_7b85aa54` FOREIGN KEY (`conversation_id`) REFERENCES `conversations` (`conversation_id`) ON DELETE CASCADE,
    KEY `idx_messages_convers_5a67d4` (`conversation_id`),
    KEY `idx_conversation_timestamp` (`conversation_id`, `timestamp`)
) CHARACTER SET utf8mb4 COMMENT='对话消息记录模型';
CREATE TABLE IF NOT EXISTS `predictions_arima_auto` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '记录ID',
    `timestamp` DATETIME(6) NOT NULL COMMENT '预测时间点',
    `point_id` VARCHAR(10) NOT NULL COMMENT '监测点ID',
    `temperature` DOUBLE COMMENT '温度',
    `pressure` DOUBLE COMMENT '压力',
    `flow_rate` DOUBLE COMMENT '流速',
    `level` DOUBLE COMMENT '液位',
    `gas_type` VARCHAR(10) COMMENT '气体类型',
    `gas_concentration` DOUBLE COMMENT '气体浓度',
    `updated_at` DATETIME(6) NOT NULL COMMENT '最后更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    KEY `idx_predictions_timesta_6ccede` (`timestamp`, `point_id`),
    KEY `idx_predictions_timesta_e157e9` (`timestamp`),
    KEY `idx_predictions_point_i_a51eca` (`point_id`)
) CHARACTER SET utf8mb4 COMMENT='ARIMA自动预测结果表模型';
CREATE TABLE IF NOT EXISTS `aerich` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `version` VARCHAR(255) NOT NULL,
    `app` VARCHAR(100) NOT NULL,
    `content` JSON NOT NULL
) CHARACTER SET utf8mb4;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """
