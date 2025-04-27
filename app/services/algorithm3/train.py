import json

# 导入TabNet相关的函数
from app.services.algorithm3.TabNet_algorithm.train_model import tableNet_train_model_and_save
# 导入xgboost相关的函数
from app.services.algorithm3.xgboost_algorithm.train_model import train_model_and_save as xgboost_train_model_and_save
# 导入lightGBM相关的函数
from app.services.algorithm3.lightGBM_algorithm.train_model import train_model_and_save as lightGBM_train_model_and_save


def load_config(config_path):
    """读取配置文件"""
    with open(config_path, "r") as config_file:
        return json.load(config_file)


def save_config(config_path, config):
    """保存配置文件"""
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)


def train_xgboost(config):
    """训练XGBoost模型"""
    print("模型训练中(xgboost)")
    xgboost_train_model_and_save(
        config,
        # 测试数据
        "Data/timeseries_data.csv",
        # 读取权重文件
        # "Data/point_id_weight/point_id_weight.csv",
        "Data/point_id_weight/monitoring_points_weights.csv",
        "Model/xgboost_model/xgboost_model_0.1_8/xgboost_model.ubj",
        "Model/xgboost_model/xgboost_model_0.1_8/label_encoder.pkl",
        "Model/xgboost_model/xgboost_model_0.1_8/feature_columns.pkl"
    )
    print("模型训练完成")


def train_lightGBM(config):
    """训练lightGBM模型"""
    print("模型训练中(lightGBM)")
    lightGBM_train_model_and_save(
        config,
        "Data/timeseries_data.csv",
        # 测试读取新的权重信息进行文件修改
        # "Data/point_id_weight.csv",
        "Data/point_id_weight/monitoring_points_weights.csv",
        "Model/lightGBM_model/lightGBM_model_0.1_4/lightgbm_model.txt",
        "Model/lightGBM_model/lightGBM_model_0.1_4/label_encoder.pkl",
        "Model/lightGBM_model/lightGBM_model_0.1_4/feature_columns.pkl"
    )
    print("模型训练完成")


def train_TabNet(config):
    """训练tableNet模型"""
    print("模型训练中(tableNet)")
    tableNet_train_model_and_save(
        config,
        "Data/timeseries_data.csv",
        # 权重文件
        "Data/point_id_weight/monitoring_points_weights.csv",
        # 模型保存
        "Model/TabNet_model/TabNet_model_0.02_50/label_encoder.pkl",
        "Model/TabNet_model/TabNet_model_0.02_50/feature_columns.pkl",
        "Model/TabNet_model/TabNet_model_0.02_50/scaler.pkl"
    )
    print("模型训练完成")


def main():
    # 读取配置文件
    config = load_config("config/config.json")

    # 检查算法类型
    if config["algorithm"] == "xgboost":
        train_xgboost(config)
        save_config("config/config.json", config)
    # 检查算法类型
    elif config["algorithm"] == "lightGBM":
        train_lightGBM(config)
        save_config("config/config.json", config)
    # 检查算法类型
    elif config["algorithm"] == "TabNet_algorithm":
        train_TabNet(config)
        save_config("config/config.json", config)
    else:
        print("不支持的算法类型")


if __name__ == "__main__":
    main()
