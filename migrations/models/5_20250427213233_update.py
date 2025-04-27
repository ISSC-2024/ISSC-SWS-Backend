from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `algorithm3_result` DROP INDEX `idx_algorithm3_result_config_point`;
        ALTER TABLE `algorithm3_result` ADD INDEX `idx_algorithm3_result_config_point` (`config_id`, `point_id`);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `algorithm3_result` DROP INDEX `idx_algorithm3_result_config_point`;
        ALTER TABLE `algorithm3_result` ADD INDEX `idx_algorithm3_result_config_point` (`point_id`, `config_id`);"""
