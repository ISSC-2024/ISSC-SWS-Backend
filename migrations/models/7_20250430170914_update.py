from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `algorithm3_result` DROP INDEX `idx_algorithm3_result_config_point`;
        CREATE TABLE IF NOT EXISTS `conversations` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `title` VARCHAR(255) NOT NULL COMMENT '对话标题',
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)
) CHARACTER SET utf8mb4 COMMENT='对话历史记录模型';
        CREATE TABLE IF NOT EXISTS `messages` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `role` VARCHAR(10) NOT NULL COMMENT '消息角色',
    `content` LONGTEXT NOT NULL COMMENT '消息内容',
    `thinking` LONGTEXT COMMENT 'AI思考过程',
    `isThinkingExpanded` BOOL COMMENT '思考过程是否展开' DEFAULT 0,
    `model` VARCHAR(50) COMMENT '使用的AI模型',
    `timestamp` DATETIME(6) NOT NULL COMMENT '消息时间' DEFAULT CURRENT_TIMESTAMP(6),
    `conversation_id` INT NOT NULL COMMENT '所属对话',
    CONSTRAINT `fk_messages_conversa_c44a3ed5` FOREIGN KEY (`conversation_id`) REFERENCES `conversations` (`id`) ON DELETE CASCADE,
    KEY `idx_messages_convers_5a67d4` (`conversation_id`),
    KEY `idx_conversation_timestamp` (`conversation_id`, `timestamp`)
) CHARACTER SET utf8mb4 COMMENT='对话消息记录模型';
        ALTER TABLE `algorithm3_result` ADD INDEX `idx_algorithm3_result_config_point` (`point_id`, `config_id`);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `algorithm3_result` DROP INDEX `idx_algorithm3_result_config_point`;
        DROP TABLE IF EXISTS `messages`;
        DROP TABLE IF EXISTS `conversations`;
        ALTER TABLE `algorithm3_result` ADD INDEX `idx_algorithm3_result_config_point` (`config_id`, `point_id`);"""
