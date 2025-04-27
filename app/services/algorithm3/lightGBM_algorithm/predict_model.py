import pandas as pd
import joblib
import logging
import lightgbm as lgb

# 设置日志记录
from app.services.algorithm3.utils.data_message import process_timeseries_data

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s)')


def load_model(model_path):
    """加载模型"""
    try:
        # 使用 LightGBM 的 Booster 类加载模型
        model = lgb.Booster(model_file=model_path)
        return model
    except Exception as e:
        logging.error(f"加载模型时发生错误：{e}")
        raise e


def preprocess_test_data(test_data_path, feature_columns):
    """预处理测试数据"""
    try:
        original_df = pd.read_csv(test_data_path)
        new_df = original_df.copy()

        if 'timestamp' in new_df.columns:
            new_df.drop('timestamp', axis=1, inplace=True)

        new_df = pd.get_dummies(new_df, columns=['gas_type', 'point_id'])

        for col in feature_columns:
            if col not in new_df.columns:
                new_df[col] = 0

        new_df = new_df[feature_columns]
        return new_df
    except Exception as e:
        logging.error(f"预处理测试数据时发生错误：{e}")
        raise e


def make_predictions(model, new_df):
    """使用模型进行预测"""
    try:
        # LightGBM 的 predict 方法返回的是概率矩阵
        probabilities = model.predict(new_df)
        # 获取每行的最大概率对应的索引（即预测的类别）
        predictions = probabilities.argmax(axis=1)
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
        original_df['risk_level'] = predicted_df['risk_level']
        original_df['risk_level_name'] = predicted_df['risk_level_name']

        original_df.to_csv(output_csv_path, index=False)
        process_timeseries_data(
            output_csv_path,
            output_csv_path,
            output_json_path
        )
        logging.info(f"预测结果已保存到CSV文件：{output_csv_path}")

        # original_df.to_json(output_json_path, orient='records', force_ascii=False)
        logging.info(f"预测结果已保存到JSON文件：{output_json_path}")
    except Exception as e:
        logging.error(f"保存结果时发生错误：{e}")
        raise e


def predict_and_save_results(model_path, label_encoder_path, feature_columns_path, test_data_path, output_csv_path, output_json_path):
    """为测试数据加标签并保存结果"""
    try:
        # 加载模型
        model = load_model(model_path)

        # 加载标签编码器
        label_encoder = joblib.load(label_encoder_path)

        # 加载特征列
        feature_columns = joblib.load(feature_columns_path)

        # 读取新的无标签数据
        new_df = preprocess_test_data(test_data_path, feature_columns)

        # 使用模型进行预测
        predictions = make_predictions(model, new_df)

        # 后处理预测结果
        predicted_df = postprocess_predictions(predictions, label_encoder)

        # 保存结果到CSV和JSON文件
        original_df = pd.read_csv(test_data_path)
        save_results(original_df, predicted_df,
                     output_csv_path, output_json_path)
    except Exception as e:
        logging.error(f"预测和保存结果时发生错误：{e}")
        raise e

# # 示例调用
# if __name__ == "__main__":
#     predict_and_save_results(
#         "../Model/lightGBM_model/lightgbm_model.txt",
#         "../Model/lightGBM_model/label_encoder.pkl",
#         "../Model/lightGBM_model/feature_columns.pkl",
#         "../Data/time_series_forecasting_data/predictions_arima_auto.csv",
#         "../Data/lightGBM/predicted_results_with_original_data.csv",
#         "../Data/lightGBM/predicted_results_with_original_data.json"
#     )
