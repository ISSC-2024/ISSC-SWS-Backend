import csv
import logging
import time
from typing import Type, List, Dict, Any, Optional, Union
from tortoise.models import Model
from pathlib import Path

logger = logging.getLogger(__name__)


async def import_csv_to_model(
    csv_path: str,
    model_class: Type[Model],
    config_id: Optional[int] = None,
    config_field_name: str = "config",
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
        config_field_name: 默认为"config"，这里应当是关系对象名，而非ID字段名
        batch_size: 批量插入的记录数量
        encoding: CSV文件编码
        custom_mapping: 自定义字段映射，覆盖自动检测的映射

    返回:
        包含导入结果统计的字典
    """
    # 记录开始时间
    start_time = time.time()

    # 检查文件是否存在
    file_path = Path(csv_path)
    if not file_path.exists():
        error_msg = f"CSV文件不存在: {csv_path}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    # 获取模型所有字段
    model_fields = model_class._meta.fields_map
    model_field_names = list(model_fields.keys())
    logger.info(f"模型字段: {model_field_names}")

    # 查找外键字段信息
    fk_field_obj = model_fields.get(config_field_name)
    fk_db_field = None

    if fk_field_obj and hasattr(fk_field_obj, 'source_field'):
        fk_db_field = fk_field_obj.source_field
        logger.info(f"找到外键数据库字段名: {fk_db_field}")

    # 获取模型的数据库字段映射
    db_to_model_fields = {}
    if hasattr(model_class._meta, 'fields_db_projection'):
        # 使用正确的字段投影属性
        fields_db_projection = model_class._meta.fields_db_projection
        # 创建反向映射: 数据库字段名 -> 模型字段名
        db_to_model_fields = {v: k for k, v in fields_db_projection.items()}
        logger.info(f"数据库字段映射: {db_to_model_fields}")

    # 统计信息
    stats = {
        "total": 0,
        "imported": 0,
        "failed": 0,
        "errors": [],
        "field_mapping": {}
    }

    # 统计文件总行数，用于显示进度
    total_lines = 0
    with open(file_path, 'r', encoding=encoding) as f:
        for _ in f:
            total_lines += 1
    # 减去标题行
    total_lines -= 1

    print(f"开始导入CSV文件: {csv_path}")
    print(f"总行数: {total_lines}")

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
                if csv_col in model_field_names:
                    field_mapping[csv_col] = csv_col
                # 检查是否匹配数据库字段名
                elif csv_col in db_to_model_fields:
                    model_field = db_to_model_fields[csv_col]
                    field_mapping[csv_col] = model_field

            # 如果没有找到可映射的字段
            if not field_mapping:
                return {"success": False, "error": "没有找到可以映射的字段"}

            # 记录使用的字段映射
            stats["field_mapping"] = field_mapping
            logger.info(f"字段映射: {field_mapping}")
            print(f"字段映射: {field_mapping}")

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
            batch_indices = []  # 记录每条记录在原始数据中的索引，用于错误定位
            batch_count = 0  # 批次计数

            for idx, row in enumerate(reader):
                stats["total"] += 1

                # 构建模型字段数据
                record_data = {}

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
                if not record_data:
                    stats["failed"] += 1
                    continue

                # 如果有配置ID，设置外键
                if config_id is not None:
                    # 使用正确的字段名设置外键ID
                    record_data[f"{config_field_name}_id"] = config_id

                batch.append(record_data)
                batch_indices.append(idx + 1)  # 记录行号（从1开始）

                # 达到批处理大小时执行批量插入
                if len(batch) >= batch_size:
                    batch_count += 1
                    try:
                        # 使用批量创建提高性能
                        await model_class.bulk_create([model_class(**item) for item in batch])
                        stats["imported"] += len(batch)

                        # 打印进度信息
                        elapsed_time = time.time() - start_time
                        progress = (stats["imported"] +
                                    stats["failed"]) / total_lines * 100
                        rate = stats["imported"] / \
                            elapsed_time if elapsed_time > 0 else 0

                        print(f"批次 {batch_count}: 已导入 {stats['imported']}/{total_lines} ({progress:.2f}%), "
                              f"速率: {rate:.1f} 记录/秒, 已用时间: {elapsed_time:.1f}秒")

                    except Exception as e:
                        # 批量创建失败时，尝试单个创建以识别问题记录
                        logger.warning(f"批量插入失败，尝试单个插入: {str(e)}")
                        print(f"批次 {batch_count} 批量插入失败，正在尝试单个插入...")

                        successful = 0
                        for i, item in enumerate(batch):
                            try:
                                await model_class.create(**item)
                                successful += 1
                            except Exception as sub_e:
                                error = f"记录 {batch_indices[i]} 导入失败: {str(sub_e)}, 数据: {item}"
                                stats["errors"].append(error)
                                logger.error(error)

                        stats["imported"] += successful
                        stats["failed"] += (len(batch) - successful)

                        # 打印单个插入的结果
                        print(
                            f"批次 {batch_count} 单个插入完成: 成功 {successful}/{len(batch)}")

                        if successful == 0:
                            # 如果所有记录插入都失败，添加总错误信息
                            error = f"批量处理完全失败: {str(e)}"
                            stats["errors"].append(error)
                            logger.error(error)

                    # 清空批处理列表
                    batch = []
                    batch_indices = []

            # 处理剩余的记录
            if batch:
                batch_count += 1
                try:
                    # 使用批量创建提高性能
                    await model_class.bulk_create([model_class(**item) for item in batch])
                    stats["imported"] += len(batch)

                    # 打印最后一批的进度
                    elapsed_time = time.time() - start_time
                    progress = 100.0  # 已完成全部
                    rate = stats["imported"] / \
                        elapsed_time if elapsed_time > 0 else 0

                    print(f"最后批次 {batch_count}: 已导入 {stats['imported']}/{total_lines} ({progress:.2f}%), "
                          f"速率: {rate:.1f} 记录/秒, 已用时间: {elapsed_time:.1f}秒")

                except Exception as e:
                    # 批量创建失败时，尝试单个创建以识别问题记录
                    logger.warning(f"批量插入失败，尝试单个插入: {str(e)}")
                    print(f"最后批次 {batch_count} 批量插入失败，正在尝试单个插入...")

                    successful = 0
                    for i, item in enumerate(batch):
                        try:
                            await model_class.create(**item)
                            successful += 1
                        except Exception as sub_e:
                            error = f"记录 {batch_indices[i]} 导入失败: {str(sub_e)}, 数据: {item}"
                            stats["errors"].append(error)
                            logger.error(error)

                    stats["imported"] += successful
                    stats["failed"] += (len(batch) - successful)

                    # 打印单个插入的结果
                    print(
                        f"最后批次 {batch_count} 单个插入完成: 成功 {successful}/{len(batch)}")

                    if successful == 0:
                        # 如果所有记录插入都失败，添加总错误信息
                        error = f"批量处理完全失败: {str(e)}"
                        stats["errors"].append(error)
                        logger.error(error)

        # 打印总结信息
        total_time = time.time() - start_time
        avg_rate = stats["imported"] / total_time if total_time > 0 else 0

        print(f"\n导入完成摘要:")
        print(f"总记录数: {stats['total']}")
        print(
            f"成功导入: {stats['imported']} ({stats['imported']/stats['total']*100:.2f}%)")
        print(
            f"导入失败: {stats['failed']} ({stats['failed']/stats['total']*100:.2f}%)")
        print(f"总用时: {total_time:.2f}秒")
        print(f"平均速率: {avg_rate:.1f} 记录/秒")

        if stats["errors"]:
            print(f"发生 {len(stats['errors'])} 个错误，详情请查看日志")

        stats["success"] = stats["failed"] < stats["total"]
        stats["total_time"] = total_time
        stats["avg_rate"] = avg_rate

        logger.info(
            f"CSV导入完成: 共{stats['total']}条记录, 成功{stats['imported']}条, 失败{stats['failed']}条, 用时{total_time:.2f}秒")
        return stats

    except Exception as e:
        error_msg = f"CSV导入过程发生错误: {str(e)}"
        logger.error(error_msg)
        print(f"导入失败: {error_msg}")
        return {"success": False, "error": error_msg}
