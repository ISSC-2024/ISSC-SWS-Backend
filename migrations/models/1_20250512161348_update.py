from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `predictions_arima_auto` DROP INDEX `idx_predictions_timesta_6ccede`;
        ALTER TABLE `predictions_arima_auto` ADD `region` VARCHAR(10) COMMENT '区域';
        ALTER TABLE `predictions_arima_auto` ADD INDEX `idx_predictions_region_de7f52` (`region`);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `predictions_arima_auto` DROP INDEX `idx_predictions_region_de7f52`;
        ALTER TABLE `predictions_arima_auto` DROP COLUMN `region`;
        ALTER TABLE `predictions_arima_auto` ADD INDEX `idx_predictions_timesta_6ccede` (`timestamp`, `point_id`);"""
