import pandas as pd
import os
import json


def calculate_region_risk(weights_file, data_file, output_dir):
    """
    计算区域风险并保存结果到 JSON 文件。

    参数:
    weights_file (str): 权重数据文件路径
    data_file (str): 监测数据文件路径
    output_dir (str): 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_file_json = os.path.join(output_dir, 'region_risk_summary.json')

    try:
        # 读取第一个文件（权重数据）
        weights_df = pd.read_csv(weights_file)

        # 读取第二个文件（监测数据）
        data_df = pd.read_csv(data_file)

        # 确保message列中的值是字符串类型，并将NaN替换为空字符串
        data_df['message'] = data_df['message'].fillna('').astype(str)

        # 创建一个字典来存储每个区域的权重总和
        region_weights = {}

        # 创建一个字典来存储每个区域的点位信息
        region_points = {}

        # 遍历权重数据，按区域前缀分组
        for _, row in weights_df.iterrows():
            point_id = row['point_id']
            weight = row['weight']
            region = point_id[:3]  # 提取前缀作为区域标识

            if region not in region_weights:
                region_weights[region] = 0.0
                region_points[region] = []

            region_weights[region] += weight
            region_points[region].append(point_id)

        # 按时间戳分组
        grouped_by_time = data_df.groupby('timestamp')

        # 用于存储所有时间戳的输出结果
        all_outputs = []

        # 遍历每个时间戳分组
        for timestamp, group in grouped_by_time:
            # 创建一个字典来存储每个区域的综合风险值
            region_risk = {}

            # 遍历监测数据，计算每个区域的综合风险值
            for _, row in group.iterrows():
                point_id = row['point_id']
                risk_level = row['risk_level']
                message = row['message']

                # 将风险等级转换为数值
                if risk_level == 'safe':
                    risk_value = 1
                elif risk_level == 'warning':
                    risk_value = 3
                else:  # danger
                    risk_value = 5

                region = point_id[:3]  # 提取区域标识

                # 查找该点位的权重
                weight_row = weights_df[weights_df['point_id'] == point_id]
                if not weight_row.empty:
                    weight = weight_row['weight'].values[0]

                    # 计算实际权重
                    actual_weight = weight / region_weights[region]

                    # 计算加权风险值
                    weighted_risk = actual_weight * risk_value

                    # 累加到区域的综合风险值
                    if region not in region_risk:
                        region_risk[region] = {'total_risk': 0.0, 'messages': []}

                    region_risk[region]['total_risk'] += weighted_risk

                    # 如果是warning或danger，且message不为空，记录信息
                    if risk_level != 'safe' and message != "":
                        region_risk[region]['messages'].append(message)

            # 输出结果
            output = []
            for region, info in region_risk.items():
                total_risk = info['total_risk']
                messages = info['messages']

                # 根据综合风险值确定区域的风险等级
                if total_risk < 1.5:
                    region_risk_level = 'safe'
                    # 如果message为空或状态是safe，设置最终message为"正常"
                    final_message = "正常" if not messages or region_risk_level == 'safe' else ''
                elif total_risk < 2.5:
                    region_risk_level = 'warning'
                    final_message = '; '.join(messages) if messages else ''
                else:
                    region_risk_level = 'danger'
                    final_message = '; '.join(messages) if messages else ''

                output.append({
                    'timestamp': timestamp,
                    'region': region,
                    'risk_level': region_risk_level,
                    'message': final_message
                })

            # 将结果转换为DataFrame
            output_df = pd.DataFrame(output)

            # 保存到所有输出中
            all_outputs.append(output_df)

        # 合并所有时间戳的输出
        if all_outputs:
            final_output_df = pd.concat(all_outputs)

            # 保存结果到 JSON 文件，确保正确处理中文字符
            final_output_df.to_json(output_file_json, orient='records', date_format='iso', force_ascii=False)
            print(f"结果已保存到 JSON 文件: {output_file_json}")
        else:
            print("没有找到任何数据。")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except pd.errors.EmptyDataError:
        print("文件为空，请检查数据文件是否正确。")
    except pd.errors.ParserError:
        print("解析文件时出错，请检查文件格式是否正确。")
    except Exception as e:
        print(f"发生错误: {e}")


# # 调用函数
# weights_file = './Data/point_id_weight/monitoring_points_weights.csv'
# data_file = './Data/predict/predict_xgboost/predicted_results_with_original_data.csv'
# output_dir = './Data/predict'
# calculate_region_risk(weights_file, data_file, output_dir)