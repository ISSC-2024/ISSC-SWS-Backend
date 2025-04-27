import os
import pandas as pd
import joblib
import numpy as np
import logging
from pytorch_tabnet.tab_model import TabNetClassifier

# 设置日志记录
from app.services.algorithm3.utils.data_message import process_timeseries_data

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_tabnet_assets(model_dir, label_encoder_path, feature_columns_path, scaler_path):
    """加载TabNet模型及相关资源（适配新路径结构）"""
    try:
        # 加载TabNet模型
        model = TabNetClassifier()
        model.load_model(os.path.join(model_dir, "TabNet_model.zip"))  # 固定文件名

        # 加载其他预处理对象
        label_encoder = joblib.load(label_encoder_path)
        feature_columns = joblib.load(feature_columns_path)
        scaler = joblib.load(scaler_path)

        logging.info(f"模型加载自目录：{model_dir}")
        return model, label_encoder, feature_columns, scaler
    except Exception as e:
        logging.error(f"资源加载失败：{e}")
        raise e


def generate_feature_weights(feature_columns, expert_weights):
    """生成特征权重数组（适配TabNet）"""
    try:
        weights = []
        for col in feature_columns:
            if col.startswith('point_id_'):
                raw_id = col.split('_', 2)[-1]
                weights.append(expert_weights.get(raw_id, 1.0))
            else:
                weights.append(1.0)
        return np.array(weights)  # 转换为numpy数组
    except Exception as e:
        logging.error(f"生成特征权重时发生错误：{e}")
        raise e


def preprocess_for_tabnet(raw_df, feature_columns, scaler, feature_weights):
    """TabNet专用数据预处理"""
    try:
        processed_df = raw_df.copy()

        # 移除时间戳（如果存在）
        if 'timestamp' in processed_df.columns:
            processed_df.drop('timestamp', axis=1, inplace=True)

        # 确保分类变量编码一致性
        processed_df = pd.get_dummies(
            processed_df, columns=['gas_type', 'point_id'])

        # 特征对齐
        for col in feature_columns:
            if col not in processed_df.columns:
                processed_df[col] = 0
        processed_df = processed_df[feature_columns]

        # 应用预处理流水线
        scaled_data = scaler.transform(processed_df)
        weighted_data = scaled_data * feature_weights  # 应用特征权重

        return weighted_data
    except Exception as e:
        logging.error(f"数据预处理失败：{e}")
        raise e


def tabnet_predict(model, processed_data):
    """执行TabNet预测"""
    try:
        predictions = model.predict(processed_data)
        logging.info(f"预测完成，样本数量：{predictions.shape[0]}")
        return predictions
    except Exception as e:
        logging.error(f"预测执行失败：{e}")
        raise e


def decode_predictions(predictions, label_encoder):
    """解码预测结果"""
    try:
        # 假设风险等级映射关系
        risk_mapping = {
            0: ('danger', '危险'),
            1: ('safe', '安全'),
            2: ('warning', '警告')
        }

        decoded = []
        for pred in predictions:
            level_code = pred
            level_name, level_cn = risk_mapping.get(
                level_code, ('unknown', '未知'))
            decoded.append({
                'risk_level': level_name,
                'risk_level_name': level_cn
            })

        return pd.DataFrame(decoded)
    except Exception as e:
        logging.error(f"结果解码失败：{e}")
        raise e


def save_results(original_df, predicted_df, output_csv_path, output_json_path):
    """保存结果到CSV和JSON文件"""
    try:
        # 将预测结果直接拼接到原始数据上
        result_df = original_df.copy()
        result_df['risk_level'] = predicted_df['risk_level']
        result_df['risk_level_name'] = predicted_df['risk_level_name']

        # 保存到CSV文件,保存到JSON文件
        result_df.to_csv(output_csv_path, index=False)
        process_timeseries_data(
            output_csv_path,
            output_csv_path,
            output_json_path
        )
        logging.info(f"CSV结果保存到：{output_csv_path}")

        # result_df.to_json(output_json_path, orient='records', force_ascii=False)
        logging.info(f"JSON结果保存到：{output_json_path}")
    except Exception as e:
        logging.error(f"保存结果时发生错误：{e}")
        raise e


def load_expert_weights(weight_path):
    """加载专家权重文件（与训练代码保持一致）"""
    try:
        weight_df = pd.read_csv(weight_path)
        return dict(zip(weight_df['point_id'], weight_df['weight']))
    except Exception as e:
        logging.error(f"加载专家权重文件失败：{e}")
        raise e


def tabnet_predict_pipeline(label_encoder_path, feature_columns_path,
                            scaler_path, test_data_path, output_csv_path,
                            output_json_path, expert_weight_path):
    """端到端预测流程（自动推导模型路径）"""
    try:
        # 自动推导模型目录
        model_dir = os.path.dirname(label_encoder_path)

        # 加载资源
        model, label_encoder, feature_columns, scaler = load_tabnet_assets(
            model_dir,
            label_encoder_path,
            feature_columns_path,
            scaler_path
        )

        # 生成特征权重
        expert_weights = load_expert_weights(expert_weight_path)
        feature_weights = generate_feature_weights(
            feature_columns, expert_weights)

        # 读取并处理数据
        raw_data = pd.read_csv(test_data_path)
        processed_data = preprocess_for_tabnet(
            raw_data, feature_columns, scaler, feature_weights)

        # 执行预测
        predictions = tabnet_predict(model, processed_data)

        # 解码结果
        decoded_df = decode_predictions(predictions, label_encoder)

        # 保存结果
        save_results(raw_data, decoded_df, output_csv_path, output_json_path)

        logging.info("预测流程成功完成！")
        return decoded_df
    except Exception as e:
        logging.error(f"预测流程异常终止：{e}")
        raise e


# # 示例调用
# if __name__ == "__main__":
#     tabnet_predict_pipeline(
#         label_encoder_path="../Model/TabNet_model/label_encoder.pkl",
#         feature_columns_path="../Model/TabNet_model/feature_columns.pkl",
#         scaler_path="../Model/TabNet_model/scaler.pkl",
#         test_data_path="../Data/time_series_forecasting_data/predictions_arima_auto.csv",
#         output_csv_path="../Data/predict/predict_tableNet/predicted_results.csv",
#         output_json_path="../Data/predict/predict_tableNet/predicted_results.json",
#         expert_weight_path="../Data/point_id_weight/monitoring_points_weights.csv"
#     )
