from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `algorithm2_result` DROP FOREIGN KEY `fk_algorith_algorith_fc7bce90`;
        ALTER TABLE `algorithm2_result` RENAME COLUMN `config_id_id` TO `config_id`;
        ALTER TABLE `algorithm2_result` ADD CONSTRAINT `fk_algorith_algorith_8f4a903a` FOREIGN KEY (`config_id`) REFERENCES `algorithm2_config` (`config_id`) ON DELETE CASCADE;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `algorithm2_result` DROP FOREIGN KEY `fk_algorith_algorith_8f4a903a`;
        ALTER TABLE `algorithm2_result` RENAME COLUMN `config_id` TO `config_id_id`;
        ALTER TABLE `algorithm2_result` ADD CONSTRAINT `fk_algorith_algorith_fc7bce90` FOREIGN KEY (`config_id_id`) REFERENCES `algorithm2_config` (`config_id`) ON DELETE CASCADE;"""
