import pandas as pd
import io
import csv
from typing import List, Dict, Any, Optional, Iterator, Union
from datetime import datetime
from fastapi import Response
from fastapi.responses import StreamingResponse


class CSVBuilder:
    """
    流式CSV构建工具 - 优化大数据量下载，适配前端blob响应类型
    """

    @staticmethod
    def _process_data(
        results: List[Any],
        column_mapping: Optional[Dict[str, str]] = None,
        date_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """处理查询结果数据为pandas DataFrame"""
        # 1. 将ORM结果转换为字典列表
        records = []
        for item in results:
            if hasattr(item, "to_dict") and callable(getattr(item, "to_dict")):
                # 使用内置to_dict方法
                records.append(item.to_dict())
            else:
                # 从__dict__中提取，过滤私有属性
                record = {}
                for k, v in item.__dict__.items():
                    if not k.startswith('_') and not callable(v):
                        record[k] = v
                records.append(record)

        # 2. 创建DataFrame
        df = pd.DataFrame(records) if records else pd.DataFrame()

        # 3. 如果数据为空，返回一个带有默认列的空DataFrame
        if df.empty:
            if column_mapping:
                df = pd.DataFrame(columns=list(column_mapping.values()))
            else:
                df = pd.DataFrame(columns=["No Data Found"])
            return df

        # 4. 排除指定列
        if exclude_columns:
            cols_to_drop = [
                col for col in exclude_columns if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

        # 5. 处理日期列
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(
                        df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

        # 6. 应用列名映射
        if column_mapping:
            rename_dict = {k: v for k,
                           v in column_mapping.items() if k in df.columns}
            if rename_dict:
                df = df.rename(columns=rename_dict)

        return df

    @staticmethod
    def _generate_csv_stream(df: pd.DataFrame, chunk_size: int = 50) -> Iterator[bytes]:
        """
        生成CSV流，分块处理避免内存过载

        参数:
            df: 要处理的DataFrame
            chunk_size: 每个块的行数

        返回:
            字节迭代器，用于流式传输
        """
        # 1. 首先发送带有BOM的UTF-8头，确保Excel正确识别编码
        yield b'\xef\xbb\xbf'

        # 2. 如果DataFrame为空，只返回表头
        if df.empty:
            header = ','.join(df.columns) + '\n'
            yield header.encode('utf-8')
            return

        # 3. 创建内存中的缓冲区
        buffer = io.StringIO()

        # 4. 写入CSV表头
        buffer.write(','.join(df.columns) + '\n')
        csv_header = buffer.getvalue().encode('utf-8')
        yield csv_header

        # 5. 清空缓冲区
        buffer.seek(0)
        buffer.truncate(0)

        # 6. 分块处理数据行
        total_rows = len(df)
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            # 获取当前块的数据
            chunk = df.iloc[start_idx:end_idx]

            # 将块写入CSV格式，不包含表头
            chunk.to_csv(buffer, index=False, header=False)

            # 获取CSV文本并转为字节
            chunk_bytes = buffer.getvalue().encode('utf-8')
            yield chunk_bytes

            # 清空缓冲区，准备下一块
            buffer.seek(0)
            buffer.truncate(0)

    @staticmethod
    def create_streaming_response(
        results: List[Any],
        filename_prefix: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        date_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> StreamingResponse:
        """
        创建流式CSV响应，优化大数据量下载

        参数:
            results: 查询结果列表
            filename_prefix: 文件名前缀
            column_mapping: 列名映射
            date_columns: 日期列名列表
            exclude_columns: 需要排除的列名列表

        返回:
            StreamingResponse: 流式响应对象
        """
        # 1. 处理数据
        df = CSVBuilder._process_data(
            results,
            column_mapping,
            date_columns,
            exclude_columns
        )

        # 2. 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix or 'export'}_{timestamp}.csv"
        # 确保文件名安全
        safe_filename = ''.join(
            c for c in filename if c.isalnum() or c in '_-.').rstrip()
        if not safe_filename.endswith('.csv'):
            safe_filename += '.csv'

        # 3. 创建流式响应
        response = StreamingResponse(
            CSVBuilder._generate_csv_stream(df),
            media_type="text/csv; charset=utf-8"
        )

        # 4. 设置响应头
        response.headers["Content-Disposition"] = f'attachment; filename="{safe_filename}"'
        return response

    @staticmethod
    def create_error_response(
        message: str,
        filename_prefix: str = "error"
    ) -> Response:
        """
        创建包含错误信息的响应

        参数:
            message: 错误信息
            filename_prefix: 文件名前缀

        返回:
            Response: 包含错误信息的CSV响应
        """
        # 创建一个简单的带有错误信息的CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{timestamp}.csv"

        # 构建CSV内容
        content = "错误信息\n" + message + "\n"

        # 创建响应
        response = Response(
            content=content.encode('utf-8'),
            media_type="text/csv; charset=utf-8"
        )
        response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'

        return response
