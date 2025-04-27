import pandas as pd
from io import StringIO
import json

def process_timeseries_data(input_file_path, output_csv_path, output_json_path):
    # 定义metadata.json中的标准值
    metadata = {
        "areas": [
            {"code": "RMS", "name": "原料储存区"},
            {"code": "REA", "name": "反应器区"},
            {"code": "SEP", "name": "分离提纯区"},
            {"code": "PRO", "name": "成品储存区"},
            {"code": "UTL", "name": "公用工程区"}
        ],
        "sensors": {
            "temperature": {
                "normal_ranges": {
                    "RMS": [15, 35],
                    "REA": [80, 150],
                    "SEP": [50, 120],
                    "PRO": [15, 35],
                    "UTL": [20, 90]
                }
            },
            "pressure": {
                "normal_ranges": {
                    "RMS": [0.1, 0.5],
                    "REA": [0.5, 3.0],
                    "SEP": [0.3, 2.0],
                    "PRO": [0.1, 0.5],
                    "UTL": [0.2, 1.5]
                }
            },
            "flow_rate": {
                "normal_ranges": {
                    "RMS": [5, 50],
                    "REA": [20, 100],
                    "SEP": [15, 90],
                    "PRO": [5, 60],
                    "UTL": [30, 150]
                }
            },
            "level": {
                "normal_ranges": {
                    "RMS": [20, 80],
                    "REA": [30, 60],
                    "SEP": [20, 70],
                    "PRO": [20, 80],
                    "UTL": [30, 70]
                }
            },
            "gas_concentration": {
                "normal_ranges": {
                    "H₂S": [0, 10],
                    "NH₃": [0, 25],
                    "CO": [0, 50]
                }
            }
        }
    }

    # 读取CSV文件
    df = pd.read_csv(input_file_path)

    # 定义一个函数，用于生成异常信息
    def generate_message(row):
        message = ""

        if row['risk_level'] == 'safe':
            area_code = row['point_id'].split('0')[0]
            message = f"安全: 区域{area_code}，传感器{row['point_id']}。"
            return message

        if row['risk_level'] in ['warning', 'danger']:
            # 提取区域代码（point_id的前缀）
            area_code = row['point_id'].split('0')[0]  # 例如，RMS001 -> RMS

            # 检查温度是否超出范围或处于临界状态
            temp_min, temp_max = metadata['sensors']['temperature']['normal_ranges'][area_code]
            temp_critical_min = temp_min + (temp_max - temp_min) * 0.25  # 临界最小值，距离下限30%
            temp_critical_max = temp_max - (temp_max - temp_min) * 0.25 # 临界最大值，距离上限30%
            if row['temperature'] > temp_max:
                message += f"温度{row['temperature']}度，超出正常范围{row['temperature'] - temp_max}度。"
            elif row['temperature'] < temp_min:
                message += f"温度{row['temperature']}度，低于正常范围{temp_min - row['temperature']}度。"
            elif row['temperature'] > temp_critical_max or row['temperature'] < temp_critical_min:
                message += f"温度{row['temperature']}度，处于临界状态。"

            # 检查压力是否超出范围或处于临界状态
            pressure_min, pressure_max = metadata['sensors']['pressure']['normal_ranges'][area_code]
            pressure_critical_min = pressure_min + (pressure_max - pressure_min) * 0.25
            pressure_critical_max = pressure_max - (pressure_max - pressure_min) * 0.25
            if row['pressure'] > pressure_max:
                message += f"压力{row['pressure']}MPa，超出正常范围{row['pressure'] - pressure_max}MPa。"
            elif row['pressure'] < pressure_min:
                message += f"压力{row['pressure']}MPa，低于正常范围{pressure_min - row['pressure']}MPa。"
            elif row['pressure'] > pressure_critical_max or row['pressure'] < pressure_critical_min:
                message += f"压力{row['pressure']}MPa，处于临界状态。"

            # 检查流量是否超出范围或处于临界状态
            flow_min, flow_max = metadata['sensors']['flow_rate']['normal_ranges'][area_code]
            flow_critical_min = flow_min + (flow_max - flow_min) * 0.25
            flow_critical_max = flow_max - (flow_max - flow_min) * 0.25
            if row['flow_rate'] > flow_max:
                message += f"流量{row['flow_rate']}m³/h，超出正常范围{row['flow_rate'] - flow_max}m³/h。"
            elif row['flow_rate'] < flow_min:
                message += f"流量{row['flow_rate']}m³/h，低于正常范围{flow_min - row['flow_rate']}m³/h。"
            elif row['flow_rate'] > flow_critical_max or row['flow_rate'] < flow_critical_min:
                message += f"流量{row['flow_rate']}m³/h，处于临界状态。"

            # 检查液位是否超出范围或处于临界状态
            level_min, level_max = metadata['sensors']['level']['normal_ranges'][area_code]
            level_critical_min = level_min + (level_max - level_min) * 0.25
            level_critical_max = level_max - (level_max - level_min) * 0.25
            if row['level'] > level_max:
                message += f"液位{row['level']}%，超出正常范围{row['level'] - level_max}%。"
            elif row['level'] < level_min:
                message += f"液位{row['level']}%，低于正常范围{level_min - row['level']}%。"
            elif row['level'] > level_critical_max or row['level'] < level_critical_min:
                message += f"液位{row['level']}%，处于临界状态。"

            # 检查气体浓度是否超出范围或处于临界状态
            gas_type = row['gas_type']
            if gas_type in metadata['sensors']['gas_concentration']['normal_ranges']:
                conc_min, conc_max = metadata['sensors']['gas_concentration']['normal_ranges'][gas_type]
                conc_critical_min = conc_min + (conc_max - conc_min) * 0.25
                conc_critical_max = conc_max - (conc_max - conc_min) * 0.25
                if row['gas_concentration'] > conc_max:
                    message += f"{gas_type}浓度{row['gas_concentration']}ppm，超出正常范围{row['gas_concentration'] - conc_max}ppm。"
                elif row['gas_concentration'] < conc_min:
                    message += f"{gas_type}浓度{row['gas_concentration']}ppm，低于正常范围{conc_min - row['gas_concentration']}ppm。"
                elif row['gas_concentration'] > conc_critical_max or row['gas_concentration'] < conc_critical_min:
                    message += f"{gas_type}浓度{row['gas_concentration']}ppm，处于临界状态。"

            # 构建最终消息
            if message:
                risk_level_name = "警告" if row['risk_level'] == 'warning' else "危险"
                # header = f"{risk_level_name}: 区域{area_code}，传感器{row['point_id']}："
                header = f"传感器{row['point_id']}："
                message = header + message
            # else:
            #     risk_level_name = "警告" if row['risk_level'] == 'warning' else "危险"
            #     message = f"传感器{row['point_id']}检测到风险"
            else:
                # 如果 message 为空，不添加任何信息
                return ""

        return message.strip() if message else ""

    # 应用generate_message函数到每一行数据
    df['message'] = df.apply(generate_message, axis=1)

    # 保存结果到CSV文件
    df.to_csv(output_csv_path, index=False)

    # 保存结果到JSON文件
    df.to_json(output_json_path, orient='records', force_ascii=False, indent=4)

# # 示例用法
# process_timeseries_data(
#     "../Data/timeseries_data.csv",
#     "Data/timeseries_data_message.csv",
#     "Data/timeseries_data_message.json"
# )