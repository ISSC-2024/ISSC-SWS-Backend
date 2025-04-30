from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `conversations` RENAME COLUMN `iconversation_idd` TO `conversation_id`;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `conversations` RENAME COLUMN `conversation_id` TO `iconversation_idd`;"""
