import pandas as pd
import joblib
import logging
import xgboost as xgb
import numpy as np

# 设置日志记录
from app.services.algorithm3.utils.data_message import process_timeseries_data

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_model_and_preprocessors(model_path, label_encoder_path, feature_columns_path):
    """加载模型和预处理对象（XGBoost专用版）"""
    try:
        # 加载XGBoost模型
        model = xgb.Booster()
        model.load_model(model_path)

        # 加载其他预处理对象
        label_encoder = joblib.load(label_encoder_path)
        feature_columns = joblib.load(feature_columns_path)

        logging.info(f"成功加载模型和预处理对象，特征列数量：{len(feature_columns)}")
        return model, label_encoder, feature_columns
    except Exception as e:
        logging.error(f"加载模型和预处理对象时发生错误：{e}")
        raise e


def preprocess_test_data(test_data_path, feature_columns):
    """预处理测试数据"""
    try:
        original_df = pd.read_csv(test_data_path)
        new_df = original_df.copy()

        if 'timestamp' in new_df.columns:
            new_df.drop('timestamp', axis=1, inplace=True)

        # 确保与训练时相同的预处理
        new_df = pd.get_dummies(new_df, columns=['gas_type', 'point_id'])

        # 特征对齐处理
        for col in feature_columns:
            if col not in new_df.columns:
                new_df[col] = 0

        # 确保列顺序一致
        new_df = new_df[feature_columns]
        return new_df
    except Exception as e:
        logging.error(f"预处理测试数据时发生错误：{e}")
        raise e


def make_predictions(model, new_df):
    """使用XGBoost模型进行预测"""
    try:
        # 转换为DMatrix格式
        dtest = xgb.DMatrix(new_df)

        # 进行预测（返回概率矩阵）
        pred_probs = model.predict(dtest)

        # 转换为类别预测
        predictions = np.argmax(pred_probs, axis=1)
        logging.info(f"预测完成，样本数量：{len(predictions)}")
        return predictions
    except Exception as e:
        logging.error(f"进行预测时发生错误：{e}")
        raise e


def postprocess_predictions(predictions, label_encoder):
    """后处理预测结果"""
    try:
        predicted_labels = label_encoder.inverse_transform(predictions)
        # 创建一个映射字典，将风险等级映射到风险等级名称（中文）
        risk_level_to_name = {label: name for label, name in zip(
            label_encoder.classes_, ['危险', '安全', '警告'])}
        predicted_df = pd.DataFrame({
            'risk_level': predicted_labels,
            'risk_level_name': pd.Series(predicted_labels).map(risk_level_to_name)
        })
        return predicted_df
    except Exception as e:
        logging.error(f"后处理预测结果时发生错误：{e}")
        raise e


def save_results(original_df, predicted_df, output_csv_path, output_json_path):
    """保存结果到CSV和JSON文件"""
    try:
        # 保留原始数据的所有列
        result_df = original_df.copy()
        result_df['risk_level'] = predicted_df['risk_level']
        result_df['risk_level_name'] = predicted_df['risk_level_name']

        # 确保时间戳列存在
        if 'timestamp' not in result_df.columns:
            result_df['timestamp'] = pd.to_datetime(
                'now').strftime('%Y-%m-%d %H:%M:%S')

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


def predict_and_save_results(model_path, label_encoder_path, feature_columns_path,
                             test_data_path, output_csv_path, output_json_path):
    """为测试数据加标签并保存结果（完整流程）"""
    try:
        # 加载模型和预处理对象
        model, label_encoder, feature_columns = load_model_and_preprocessors(
            model_path, label_encoder_path, feature_columns_path
        )

        # 读取并预处理数据
        new_df = preprocess_test_data(test_data_path, feature_columns)

        # 进行预测
        predictions = make_predictions(model, new_df)

        # 后处理结果
        predicted_df = postprocess_predictions(predictions, label_encoder)

        # 保存结果
        original_df = pd.read_csv(test_data_path)
        save_results(original_df, predicted_df,
                     output_csv_path, output_json_path)

        logging.info("完整流程执行成功！")
    except Exception as e:
        logging.error(f"预测流程失败：{e}")
        raise e


# # 示例调用（注意模型扩展名）
# if __name__ == "__main__":
#     predict_and_save_results(
#         "../Model/xgboost_model/xgboost_model.ubj",
#         "../Model/xgboost_model/label_encoder.pkl",
#         "../Model/xgboost_model/feature_columns.pkl",
#         "../Data/time_series_forecasting_data/predictions_arima_auto.csv.csv",
#         "../Data/xgboost/predicted_results_with_original_data.csv",
#         "../Data/xgboost/predicted_results_with_original_data.json"
#     )
