from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `algorithm3_result` DROP INDEX `idx_algorithm3_result_config_point`;
        ALTER TABLE `algorithm3_result` MODIFY COLUMN `message` LONGTEXT COMMENT '风险描述信息';
        ALTER TABLE `algorithm3_result` ADD INDEX `idx_algorithm3_result_config_point` (`point_id`, `config_id`);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `algorithm3_result` DROP INDEX `idx_algorithm3_result_config_point`;
        ALTER TABLE `algorithm3_result` MODIFY COLUMN `message` LONGTEXT NOT NULL COMMENT '风险描述信息';
        ALTER TABLE `algorithm3_result` ADD INDEX `idx_algorithm3_result_config_point` (`config_id`, `point_id`);"""
