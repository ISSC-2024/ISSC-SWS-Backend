import pandas as pd
import joblib
import numpy as np
import torch
import json
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.utils.class_weight import compute_class_weight


def load_expert_weights(weight_path):
    """加载专家权重文件（保持不变）"""
    try:
        weight_df = pd.read_csv(weight_path)
        return dict(zip(weight_df['point_id'], weight_df['weight']))
    except Exception as e:
        print(f"加载专家权重文件时发生错误：{e}")
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
        print(f"生成特征权重时发生错误：{e}")
        raise e


def load_and_preprocess_data(data_path, weight_path):
    """加载数据并进行预处理（修复特征维度问题）"""
    try:
        expert_weights = load_expert_weights(weight_path)
        df = pd.read_csv(data_path)
        df.drop('timestamp', axis=1, inplace=True)

        # 标签编码
        label_encoder = LabelEncoder()
        df['risk_level'] = label_encoder.fit_transform(df['risk_level'])

        # 分类变量处理
        df = pd.get_dummies(df, columns=['gas_type', 'point_id'])

        # 分离特征和标签
        X = df.drop(['risk_level', 'risk_level_name'], axis=1)
        y_df = df[['risk_level', 'risk_level_name']]

        # 生成特征权重
        feature_weights = generate_feature_weights(X.columns.tolist(), expert_weights)

        # 验证维度一致性
        assert len(feature_weights) == X.shape[1], \
            f"特征权重维度({len(feature_weights)})与特征数量({X.shape[1]})不匹配"

        return X, y_df, label_encoder, feature_weights
    except Exception as e:
        print(f"加载和预处理数据时发生错误：{e}")
        raise e


def train_tabnet_model(X_train, y_train, config, feature_weights):
    """训练TabNet模型"""
    try:
        # 标准化处理
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # 类别权重计算
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights_dict = {i: w for i, w in enumerate(class_weights)}

        # 定义TabNet模型
        tabnet_params = {
            'n_d': config['n_d'],  # 特征处理维度
            'n_a': config['n_a'],  # 注意力维度
            'n_steps': config['n_steps'],  # 决策步骤
            'gamma': config['gamma'],  # 正则化系数
            'lambda_sparse': config.get('lambda_sparse', 1e-3),  # 稀疏性正则化强度
            'optimizer_params': {'lr': config['learning_rate']},
            'scheduler_params': {'step_size': 10, 'gamma': 0.9},
            'mask_type': 'sparsemax',  # 特征选择方式
            'device_name': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        clf = TabNetClassifier(**tabnet_params)

        # 应用特征权重到输入数据（通过特征缩放实现）
        X_weighted = X_train_scaled * feature_weights

        # 训练模型
        clf.fit(
            X_weighted, y_train.values,
            eval_set=[(X_weighted, y_train.values)],  # 可添加验证集
            eval_name=['train'],
            eval_metric=['accuracy'],
            max_epochs=config['max_epochs'],
            patience=15,
            batch_size=256,
            virtual_batch_size=128,
            weights=class_weights_dict,
            drop_last=False
        )

        return clf, scaler
    except Exception as e:
        print(f"训练模型时发生错误：{e}")
        raise e


def tableNet_train_model_and_save(config, data_path, weight_path,
                         label_encoder_path, feature_columns_path, scaler_path):
    """主训练流程"""
    try:
        # 设置默认参数
        final_config = {
            "n_d": 64,
            "n_a": 32,
            "n_steps": 5,
            "gamma": 1.5,
            "lambda_sparse": 1e-3,
            "max_epochs": config["max_epochs"],
            "patience": 15,
            "batch_size": 256,
            "learning_rate": config["learning_rate"],
            "random_state": 42
        }

        # 修改后的数据加载
        X, y_df, label_encoder, feature_weights = load_and_preprocess_data(data_path, weight_path)
        y = y_df['risk_level']

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=final_config["random_state"],
            stratify=y
        )

        # 训练模型
        model, scaler = train_tabnet_model(X_train, y_train, final_config, feature_weights)

        # 评估模型
        X_test_scaled = scaler.transform(X_test)
        X_test_weighted = X_test_scaled * feature_weights
        y_pred = model.predict(X_test_weighted)

        # 输出评估结果
        class_names = label_encoder.classes_
        accuracy = accuracy_score(y_test, y_pred)
        precisions = precision_score(y_test, y_pred, average=None)

        print("\n评估结果：")
        print(f"整体准确率：{accuracy:.4f}")
        print("\n各类别精确度：")
        for i, name in enumerate(class_names):
            print(f"{name}: {precisions[i]:.4f}")

        print("\n详细分类报告：")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # 自动生成模型保存路径
        import os
        model_dir = os.path.dirname(label_encoder_path)
        os.makedirs(model_dir, exist_ok=True)

        # 保存模型及元数据
        model.save_model(os.path.join(model_dir, "TabNet_model"))
        joblib.dump(label_encoder, label_encoder_path)
        joblib.dump(X.columns.tolist(), feature_columns_path)
        joblib.dump(scaler, scaler_path)

        print(f"模型及元数据已保存至目录：{model_dir}")
    except Exception as e:
        print(f"训练模型并保存时发生错误：{e}")
        raise e

# if __name__ == "__main__":
#     # 配置文件路径
#     config_path = "../config/config.json"
#
#     # 从配置文件加载参数
#     try:
#         with open(config_path, 'r') as f:
#             tabnet_config = json.load(f)
#     except FileNotFoundError:
#         print(f"配置文件 {config_path} 不存在，请检查路径是否正确。")
#         exit(1)
#     except json.JSONDecodeError:
#         print(f"配置文件 {config_path} 格式不正确，请检查文件内容。")
#         exit(1)
#
#     # 调用训练函数
#     tableNet_train_model_and_save(
#         tabnet_config,
#         "../Data/timeseries_data.csv",
#         "../Data/point_id_weight/monitoring_points_weights.csv",
#         "../Model/TabNet_model/label_encoder.pkl",
#         "../Model/TabNet_model/feature_columns.pkl",
#         "../Model/TabNet_model/scaler.pkl"
#     )