import json
# 导入TabNet相关的函数
from app.services.algorithm3.TabNet_algorithm.predict_model import tabnet_predict_pipeline
# 导入xgboost相关的函数
from app.services.algorithm3.xgboost_algorithm.predict_model import predict_and_save_results as xgboost_predict_and_save_results
# 导入lightGBM相关的函数
from app.services.algorithm3.lightGBM_algorithm.predict_model import predict_and_save_results as lightGBM_predict_and_save_results
# 导入区域风险综合
from app.services.algorithm3.utils.calculate_risk_message import calculate_region_risk


def load_config(config_path):
    """读取配置文件"""
    with open(config_path, "r") as config_file:
        return json.load(config_file)


def save_config(config_path, config):
    """保存配置文件"""
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)


def predict_xgboost(config):
    """使用XGBoost模型进行预测"""
    print("化工数据预测中(xgboost)")
    if config["learning_rate"] == 0.1 and config["max_depth"] == 6:
        xgboost_predict_and_save_results(
            "Model/xgboost_model/xgboost_model_0.1_6/xgboost_model.ubj",
            "Model/xgboost_model/xgboost_model_0.1_6/label_encoder.pkl",
            "Model/xgboost_model/xgboost_model_0.1_6/feature_columns.pkl",
            # 读取时序预测数据进行分类
            "Data/time_series_forecasting_data/predictions_arima_auto.csv",
            # 数据统一保存
            "Data/predict/predict_xgboost/xgboost_model_0.1_6/predicted_results.csv",
            "Data/predict/predict_xgboost/xgboost_model_0.1_6/predicted_results.json"
        )
        # 预测数据后调用calculate_region_risk函数
        print("区域统一信息中")
        weights_file = './Data/point_id_weight/monitoring_points_weights.csv'
        data_file = './Data/predict/predict_xgboost/xgboost_model_0.1_6/predicted_results.csv'
        output_dir = './Data/predict/predict_xgboost/xgboost_model_0.1_6'
        calculate_region_risk(weights_file, data_file, output_dir)
    elif config["learning_rate"] == 0.1 and config["max_depth"] == 8:
        xgboost_predict_and_save_results(
            "Model/xgboost_model/xgboost_model_0.1_8/xgboost_model.ubj",
            "Model/xgboost_model/xgboost_model_0.1_8/label_encoder.pkl",
            "Model/xgboost_model/xgboost_model_0.1_8/feature_columns.pkl",
            # 读取时序预测数据进行分类
            "Data/time_series_forecasting_data/predictions_arima_auto.csv",
            # 数据统一保存
            "Data/predict/predict_xgboost/xgboost_model_0.1_8/predicted_results.csv",
            "Data/predict/predict_xgboost/xgboost_model_0.1_8/predicted_results.json"
        )
        # 预测数据后调用calculate_region_risk函数
        print("区域统一信息中")
        weights_file = './Data/point_id_weight/monitoring_points_weights.csv'
        data_file = './Data/predict/predict_xgboost/xgboost_model_0.1_8/predicted_results.csv'
        output_dir = './Data/predict/predict_xgboost/xgboost_model_0.1_8'
        calculate_region_risk(weights_file, data_file, output_dir)
    else:
        print("不存在对应模型")
    print("预测完成")


def predict_lightGBM(config):
    """使用lightGBM模型进行预测"""
    print("化工数据预测中(lightGBM)")
    if config["learning_rate"] == 0.1 and config["max_depth"] == 6:
        lightGBM_predict_and_save_results(
            "Model/lightGBM_model/lightGBM_model_0.1_6/lightgbm_model.txt",
            "Model/lightGBM_model/lightGBM_model_0.1_6/label_encoder.pkl",
            "Model/lightGBM_model/lightGBM_model_0.1_6/feature_columns.pkl",
            # 读取时序预测数据进行分类
            "Data/time_series_forecasting_data/predictions_arima_auto.csv",
            # 数据统一保存
            "Data/predict/predict_lightGBM/lightGBM_model_0.1_6/predicted_results.csv",
            "Data/predict/predict_lightGBM/lightGBM_model_0.1_6/predicted_results.json"
        )
        # 预测数据后调用calculate_region_risk函数
        print("区域统一信息中")
        weights_file = './Data/point_id_weight/monitoring_points_weights.csv'
        data_file = './Data/predict/predict_lightGBM/lightGBM_model_0.1_6/predicted_results.csv'
        output_dir = './Data/predict/predict_lightGBM/lightGBM_model_0.1_6'
        calculate_region_risk(weights_file, data_file, output_dir)
    elif config["learning_rate"] == 0.1 and config["max_depth"] == 4:
        lightGBM_predict_and_save_results(
            "Model/lightGBM_model/lightGBM_model_0.1_4/lightgbm_model.txt",
            "Model/lightGBM_model/lightGBM_model_0.1_4/label_encoder.pkl",
            "Model/lightGBM_model/lightGBM_model_0.1_4/feature_columns.pkl",
            # 读取时序预测数据进行分类
            "Data/time_series_forecasting_data/predictions_arima_auto.csv",
            # 数据统一保存
            "Data/predict/predict_lightGBM/lightGBM_model_0.1_4/predicted_results.csv",
            "Data/predict/predict_lightGBM/lightGBM_model_0.1_4/predicted_results.json"
        )
        # 预测数据后调用calculate_region_risk函数
        print("区域统一信息中")
        weights_file = './Data/point_id_weight/monitoring_points_weights.csv'
        data_file = './Data/predict/predict_lightGBM/lightGBM_model_0.1_4/predicted_results.csv'
        output_dir = './Data/predict/predict_lightGBM/lightGBM_model_0.1_4'
        calculate_region_risk(weights_file, data_file, output_dir)
    else:
        print("不存在对应模型")
    print("预测完成")


def predict_tableNet(config):
    """使用tableNet模型进行预测"""
    print("化工数据预测中(tableNet)")
    if config["learning_rate"] == 0.01 and config["max_epochs"] == 100:
        tabnet_predict_pipeline(
            "Model/TabNet_model/TabNet_model_0.01_100/label_encoder.pkl",
            "Model/TabNet_model/TabNet_model_0.01_100/feature_columns.pkl",
            "Model/TabNet_model/TabNet_model_0.01_100/scaler.pkl",
            # 预测数据
            "Data/time_series_forecasting_data/predictions_arima_auto.csv",
            # 数据统一保存
            "Data/predict/predict_tableNet/TabNet_model_0.01_100/predicted_results.csv",
            "Data/predict/predict_tableNet/TabNet_model_0.01_100/predicted_results.json",
            # 权重文件
            "Data/point_id_weight/monitoring_points_weights.csv"
        )
        print("区域统一信息中")
        weights_file = './Data/point_id_weight/monitoring_points_weights.csv'
        data_file = './Data/predict/predict_tableNet/TabNet_model_0.01_100/predicted_results.csv'
        output_dir = './Data/predict/predict_tableNet/TabNet_model_0.01_100'
        calculate_region_risk(weights_file, data_file, output_dir)
    elif config["learning_rate"] == 0.02 and config["max_epochs"] == 50:
        tabnet_predict_pipeline(
            "Model/TabNet_model/TabNet_model_0.02_50/label_encoder.pkl",
            "Model/TabNet_model/TabNet_model_0.02_50/feature_columns.pkl",
            "Model/TabNet_model/TabNet_model_0.02_50/scaler.pkl",
            # 预测数据
            "Data/time_series_forecasting_data/predictions_arima_auto.csv",
            # 数据统一保存
            "Data/predict/predict_tableNet/TabNet_model_0.02_50/predicted_results.csv",
            "Data/predict/predict_tableNet/TabNet_model_0.02_50/predicted_results.json",
            # 权重文件
            "Data/point_id_weight/monitoring_points_weights.csv"
        )
        print("区域统一信息中")
        weights_file = './Data/point_id_weight/monitoring_points_weights.csv'
        data_file = './Data/predict/predict_tableNet/TabNet_model_0.02_50/predicted_results.csv'
        output_dir = './Data/predict/predict_tableNet/TabNet_model_0.02_50'
        calculate_region_risk(weights_file, data_file, output_dir)
    else:
        print("不存在对应模型")
    print("预测完成")


def main():
    # 读取配置文件
    config = load_config("config/config.json")

    # 检查算法类型
    if config["algorithm"] == "xgboost":
        predict_xgboost(config)
    # 检查算法类型
    elif config["algorithm"] == "lightGBM":
        predict_lightGBM(config)
    # 检查算法类型
    elif config["algorithm"] == "TabNet":
        predict_tableNet(config)
    else:
        print("不支持的算法类型")


if __name__ == "__main__":
    main()
