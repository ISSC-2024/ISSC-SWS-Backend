# import json
#
# # 导入xgboost相关的函数
# import time
# import datetime
#
# from Classification.TabNet_algorithm.predict_model import tabnet_predict_pipeline
# from Classification.TabNet_algorithm.train_model import tableNet_train_model_and_save
# from Classification.utils.calculate_risk_message import calculate_region_risk
# from xgboost_algorithm.predict_model import predict_and_save_results as xgboost_predict_and_save_results
# from xgboost_algorithm.train_model import train_model_and_save as xgboost_train_model_and_save
#
# # 导入lightGBM相关的函数
# from lightGBM_algorithm.predict_model import predict_and_save_results as lightGBM_predict_and_save_results
# from lightGBM_algorithm.train_model import train_model_and_save as lightGBM_train_model_and_save
#
#
# def load_config(config_path):
#     """读取配置文件"""
#     with open(config_path, "r") as config_file:
#         return json.load(config_file)
#
#
# def save_config(config_path, config):
#     """保存配置文件"""
#     with open(config_path, "w") as config_file:
#         json.dump(config, config_file, indent=4)
#
#
# def train_xgboost(config):
#     """训练XGBoost模型"""
#     print("模型训练中(xgboost)")
#     xgboost_train_model_and_save(
#         config,
#         # 测试数据
#         "Data/timeseries_data.csv",
#         # 读取权重文件
#         # "Data/point_id_weight/point_id_weight.csv",
#         "Data/point_id_weight/monitoring_points_weights.csv",
#         "Model/xgboost_model/xgboost_model.ubj",
#         "Model/xgboost_model/label_encoder.pkl",
#         "Model/xgboost_model/feature_columns.pkl"
#     )
#     print("模型训练完成")
#
#
# def predict_xgboost(config):
#     """使用XGBoost模型进行预测"""
#     print("化工数据预测中(xgboost)")
#     xgboost_predict_and_save_results(
#         "Model/xgboost_model/xgboost_model.ubj",
#         "Model/xgboost_model/label_encoder.pkl",
#         "Model/xgboost_model/feature_columns.pkl",
#         # 读取时序预测数据进行分类
#         "Data/time_series_forecasting_data/predictions_arima_auto.csv",
#         # 测试读取原数据看看结果
#         # "Data/timeseries_data_test.csv",
#         # "Data/predict/predict_xgboost/predicted_results_with_original_data.csv",
#         # "Data/predict/predict_xgboost/predicted_results_with_original_data.json"
#         # 数据统一保存
#         "Data/predict/final_result/predicted_results.csv",
#         "Data/predict/final_result/predicted_results.json"
#     )
#     print("预测完成")
#
#
# def train_lightGBM(config):
#     """训练lightGBM模型"""
#     print("模型训练中(lightGBM)")
#     lightGBM_train_model_and_save(
#         config,
#         "Data/timeseries_data.csv",
#         # 测试读取新的权重信息进行文件修改
#         # "Data/point_id_weight.csv",
#         "Data/point_id_weight/monitoring_points_weights.csv",
#         "Model/lightGBM_model/lightgbm_model.txt",
#         "Model/lightGBM_model/label_encoder.pkl",
#         "Model/lightGBM_model/feature_columns.pkl"
#     )
#     print("模型训练完成")
#
#
# def predict_lightGBM(config):
#     """使用lightGBM模型进行预测"""
#     print("化工数据预测中(lightGBM)")
#     lightGBM_predict_and_save_results(
#         "Model/lightGBM_model/lightgbm_model.txt",
#         "Model/lightGBM_model/label_encoder.pkl",
#         "Model/lightGBM_model/feature_columns.pkl",
#         # 测试读取新的权重信息进行文件修改
#         # "Data/timeseries_data_test.csv",
#         # 读取时序预测数据进行分类
#         "Data/time_series_forecasting_data/predictions_arima_auto.csv",
#         # "Data/predict/predict_lightGBM/monitoring_points_weights_predicted_results_with_original_data.csv",
#         # "Data/predict/predict_lightGBM/monitoring_points_weights_predicted_results_with_original_data.json"
#         # 数据统一保存
#         "Data/predict/final_result/predicted_results.csv",
#         "Data/predict/final_result/predicted_results.json"
#     )
#     print("预测完成")
#
# def train_tableNet(config):
#     """训练tableNet模型"""
#     print("模型训练中(tableNet)")
#     tableNet_train_model_and_save(
#         config,
#         "Data/timeseries_data.csv",
#         # 权重文件
#         "Data/point_id_weight/monitoring_points_weights.csv",
#         # 模型保存
#         "Model/TabNet_model/label_encoder.pkl",
#         "Model/TabNet_model/feature_columns.pkl",
#         "Model/TabNet_model/scaler.pkl"
#     )
#     print("模型训练完成")
#
# def predict_tableNet(config):
#     """使用tableNet模型进行预测"""
#     print("化工数据预测中(tableNet)")
#     tabnet_predict_pipeline(
#         "Model/TabNet_model/label_encoder.pkl",
#         "Model/TabNet_model/feature_columns.pkl",
#         "Model/TabNet_model/scaler.pkl",
#         # 预测数据
#         "Data/time_series_forecasting_data/predictions_arima_auto.csv",
#         # 数据统一保存
#         "Data/predict/final_result/predicted_results.csv",
#         "Data/predict/final_result/predicted_results.json",
#         # 权重文件
#         "Data/point_id_weight/monitoring_points_weights.csv"
#     )
#     print("预测完成")
#
# def main():
#     # 读取配置文件
#     config = load_config("config/config.json")
#     # 读取时间文件
#     algorithm_time_consuming = load_config("Data/predict/final_result/algorithm_time_consuming.json")
#
#     # 检查算法类型
#     if config["algorithm"] == "xgboost":
#         if config["isChanged"] == 1:
#             start_time = time.time()  # 开始时间
#             train_xgboost(config)
#             end_time = time.time()  # 结束时间
#             training_time = end_time - start_time  # 预测耗时
#             print("xgboost训练耗时:",training_time)
#             algorithm_time_consuming["training_time"] = training_time
#             algorithm_time_consuming["prediction_time"] = 0
#             save_config("Data/predict/final_result/algorithm_time_consuming.json", algorithm_time_consuming)
#             # 训练完成后，将isChanged设置为0
#             config["isChanged"] = 0
#             save_config("config/config.json", config)
#         else:
#             start_time = time.time()  # 开始时间
#             predict_xgboost(config)
#             end_time = time.time()  # 结束时间
#             prediction_time = end_time - start_time  # 预测耗时
#             print("xgboost预测耗时:", prediction_time)
#             algorithm_time_consuming["prediction_time"] = prediction_time
#             save_config("Data/predict/final_result/algorithm_time_consuming.json", algorithm_time_consuming)
#             # 预测数据后调用calculate_region_risk函数
#             print("区域统一信息中")
#             weights_file = './Data/point_id_weight/monitoring_points_weights.csv'
#             data_file = './Data/predict/final_result/predicted_results.csv'
#             output_dir = './Data/predict/final_result'
#             calculate_region_risk(weights_file, data_file, output_dir)
#     # 检查算法类型
#     elif config["algorithm"] == "lightGBM":
#         if config["isChanged"] == 1:
#             start_time = time.time()  # 开始时间
#             train_lightGBM(config)
#             end_time = time.time()  # 结束时间
#             training_time = end_time - start_time  # 预测耗时
#             print("lightGBM训练耗时:", training_time)
#             algorithm_time_consuming["training_time"] = training_time
#             algorithm_time_consuming["prediction_time"] = 0
#             save_config("Data/predict/final_result/algorithm_time_consuming.json", algorithm_time_consuming)
#             # 训练完成后，将isChanged设置为0
#             config["isChanged"] = 0
#             save_config("config/config.json", config)
#         else:
#             start_time = time.time()  # 开始时间
#             predict_lightGBM(config)
#             end_time = time.time()  # 结束时间
#             prediction_time = end_time - start_time  # 预测耗时
#             algorithm_time_consuming["prediction_time"] = prediction_time
#             print("lightGBM预测耗时:", prediction_time)
#             save_config("Data/predict/final_result/algorithm_time_consuming.json", algorithm_time_consuming)
#             # 预测数据后调用calculate_region_risk函数
#             print("区域统一信息中")
#             weights_file = './Data/point_id_weight/monitoring_points_weights.csv'
#             data_file = './Data/predict/final_result/predicted_results.csv'
#             output_dir = './Data/predict/final_result'
#             calculate_region_risk(weights_file, data_file, output_dir)
#             # 检查算法类型
#     elif config["algorithm"] == "tableNet":
#         if config["isChanged"] == 1:
#             start_time = time.time()  # 开始时间
#             train_tableNet(config)
#             end_time = time.time()  # 结束时间
#             training_time = end_time - start_time  # 预测耗时
#             print("tableNet训练耗时:", training_time)
#             algorithm_time_consuming["training_time"] = training_time
#             algorithm_time_consuming["prediction_time"] = 0
#             save_config("Data/predict/final_result/algorithm_time_consuming.json", algorithm_time_consuming)
#             # 训练完成后，将isChanged设置为0
#             config["isChanged"] = 0
#             save_config("config/config.json", config)
#         else:
#             start_time = time.time()  # 开始时间
#             predict_tableNet(config)
#             end_time = time.time()  # 结束时间
#             prediction_time = end_time - start_time  # 预测耗时
#             algorithm_time_consuming["prediction_time"] = prediction_time
#             print("tableNet预测耗时:", prediction_time)
#             save_config("Data/predict/final_result/algorithm_time_consuming.json", algorithm_time_consuming)
#             # 预测数据后调用calculate_region_risk函数
#             print("区域统一信息中")
#             weights_file = './Data/point_id_weight/monitoring_points_weights.csv'
#             data_file = './Data/predict/final_result/predicted_results.csv'
#             output_dir = './Data/predict/final_result'
#             calculate_region_risk(weights_file, data_file, output_dir)
#     else:
#         print("不支持的算法类型")
#
#
# if __name__ == "__main__":
#     main()
