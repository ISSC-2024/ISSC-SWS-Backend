import csv
import logging
import inspect
from typing import Type, List, Dict, Any, Optional, Union
from tortoise.models import Model
from tortoise.fields import Field
from pathlib import Path

logger = logging.getLogger(__name__)


async def import_csv_to_model(
    csv_path: str,
    model_class: Type[Model],
    config_id: Optional[int] = None,
    config_field_name: str = "config_id",
    batch_size: int = 100,
    encoding: str = 'utf-8',
    custom_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    通用CSV数据导入工具 - 从CSV文件导入数据到任意Tortoise ORM模型

    参数:
        csv_path: CSV文件路径
        model_class: Tortoise ORM模型类
        config_id: 可选的配置ID(外键)，如果提供则会设置到相应字段
        config_field_name: 配置ID字段名称，默认为"config_id"
        batch_size: 批量插入的记录数量
        encoding: CSV文件编码
        custom_mapping: 自定义字段映射，覆盖自动检测的映射

    返回:
        包含导入结果统计的字典
    """
    # 检查文件是否存在
    file_path = Path(csv_path)
    if not file_path.exists():
        error_msg = f"CSV文件不存在: {csv_path}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    # 获取模型所有字段
    model_fields = {}
    for name, field in model_class.__dict__.items():
        if isinstance(field, Field):
            model_fields[name] = field

    # 统计信息
    stats = {
        "total": 0,
        "imported": 0,
        "failed": 0,
        "errors": [],
        "field_mapping": {}
    }

    try:
        with open(file_path, 'r', encoding=encoding) as csvfile:
            # 跳过CSV中的注释行
            while True:
                pos = csvfile.tell()
                line = csvfile.readline()
                if not line or not line.startswith('//'):
                    csvfile.seek(pos)
                    break

            # 读取CSV头部获取列名
            reader = csv.DictReader(csvfile)
            csv_columns = reader.fieldnames if reader.fieldnames else []

            if not csv_columns:
                return {"success": False, "error": "CSV文件没有列名"}

            # 生成字段映射（CSV列名 -> 模型字段名）
            field_mapping = {}

            # 1. 首先使用自定义映射
            if custom_mapping:
                field_mapping.update(custom_mapping)

            # 2. 然后添加自动检测的同名字段
            for csv_col in csv_columns:
                # 如果已经在自定义映射中，跳过
                if csv_col in field_mapping:
                    continue

                # 检查CSV列名是否直接匹配模型字段
                if csv_col in model_fields:
                    field_mapping[csv_col] = csv_col

            # 如果没有找到可映射的字段
            if not field_mapping:
                return {"success": False, "error": "没有找到可以映射的字段"}

            # 记录使用的字段映射
            stats["field_mapping"] = field_mapping

            # 重置文件指针，重新开始读取
            csvfile.seek(0)
            # 跳过注释行
            while True:
                pos = csvfile.tell()
                line = csvfile.readline()
                if not line or not line.startswith('//'):
                    csvfile.seek(pos)
                    break

            reader = csv.DictReader(csvfile)
            batch = []

            for row in reader:
                stats["total"] += 1

                # 构建模型字段数据
                record_data = {}

                # 设置配置ID（如果提供）
                if config_id is not None:
                    # 处理外键字段命名约定
                    if config_field_name.endswith("_id"):
                        record_data[config_field_name] = config_id
                    else:
                        record_data[f"{config_field_name}_id"] = config_id

                # 根据映射填充数据
                for csv_field, model_field in field_mapping.items():
                    if csv_field in row and row[csv_field]:
                        # 获取字段类型以进行适当的转换
                        field_obj = model_fields.get(model_field)

                        if field_obj:
                            # 根据字段类型转换值
                            try:
                                if hasattr(field_obj, "field_type"):
                                    # IntField, FloatField等
                                    if field_obj.field_type == "INT":
                                        record_data[model_field] = int(
                                            row[csv_field])
                                    elif field_obj.field_type == "FLOAT" or field_obj.field_type == "DECIMAL":
                                        record_data[model_field] = float(
                                            row[csv_field])
                                    elif field_obj.field_type == "BOOL":
                                        value = row[csv_field].lower()
                                        record_data[model_field] = value in (
                                            "true", "yes", "1", "t", "y")
                                    else:  # 默认为字符串
                                        record_data[model_field] = row[csv_field]
                                else:
                                    # 默认为字符串
                                    record_data[model_field] = row[csv_field]
                            except (ValueError, TypeError) as e:
                                # 转换失败，记录错误但继续处理
                                error = f"字段转换错误 - 行 {stats['total']}, 字段 {csv_field}: {str(e)}"
                                stats["errors"].append(error)
                                logger.warning(error)

                # 如果没有有效字段，跳过此行
                if not any(key != f"{config_field_name}_id" for key in record_data.keys()):
                    stats["failed"] += 1
                    continue

                batch.append(record_data)

                # 达到批处理大小时执行批量插入
                if len(batch) >= batch_size:
                    try:
                        await model_class.bulk_create(
                            [model_class(**item) for item in batch]
                        )
                        stats["imported"] += len(batch)
                    except Exception as e:
                        stats["failed"] += len(batch)
                        error = f"批量导入失败 (记录 {stats['total'] - len(batch) + 1} - {stats['total']}): {str(e)}"
                        stats["errors"].append(error)
                        logger.error(error)

                    # 清空批处理列表
                    batch = []

            # 处理剩余的记录
            if batch:
                try:
                    await model_class.bulk_create(
                        [model_class(**item) for item in batch]
                    )
                    stats["imported"] += len(batch)
                except Exception as e:
                    stats["failed"] += len(batch)
                    error = f"批量导入失败 (剩余记录): {str(e)}"
                    stats["errors"].append(error)
                    logger.error(error)

        stats["success"] = stats["failed"] < stats["total"]
        logger.info(
            f"CSV导入完成: 共{stats['total']}条记录, 成功{stats['imported']}条, 失败{stats['failed']}条")
        return stats

    except Exception as e:
        error_msg = f"CSV导入过程发生错误: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
