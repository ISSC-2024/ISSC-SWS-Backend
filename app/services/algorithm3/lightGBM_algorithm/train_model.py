import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, classification_report
import lightgbm as lgb


def load_expert_weights(weight_path):
    """加载专家权重文件（保持不变）"""
    try:
        weight_df = pd.read_csv(weight_path)
        return dict(zip(weight_df['point_id'], weight_df['weight']))
    except Exception as e:
        print(f"加载专家权重文件时发生错误：{e}")
        raise e


def generate_sample_weights(df, expert_weights):
    """生成样本权重数组"""
    try:
        sample_weights = []
        for index, row in df.iterrows():
            point_id = row['point_id']
            sample_weights.append(expert_weights.get(point_id, 1.0))
        return sample_weights
    except Exception as e:
        print(f"生成样本权重时发生错误：{e}")
        raise e


def load_and_preprocess_data(data_path, weight_path):
    """加载数据并进行预处理（关键修改）"""
    try:
        expert_weights = load_expert_weights(weight_path)
        df = pd.read_csv(data_path)
        df.drop('timestamp', axis=1, inplace=True)

        # 编码标签列
        label_encoder = LabelEncoder()
        df['risk_level'] = label_encoder.fit_transform(df['risk_level'])

        # 生成样本权重
        sample_weights = generate_sample_weights(df, expert_weights)

        # 生成虚拟变量
        df = pd.get_dummies(df, columns=['gas_type', 'point_id'])
        X = df.drop(['risk_level', 'risk_level_name'], axis=1)

        return X, df['risk_level'], label_encoder, sample_weights
    except Exception as e:
        print(f"加载和预处理数据时发生错误：{e}")
        raise e


def train_lightgbm_model(X_train, y_train, sample_weights, config):
    """训练 LightGBM 模型（修正特征权重应用位置）"""
    try:
        # 转换为LightGBM数据集
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            weight=sample_weights,  # 使用样本权重
            free_raw_data=False
        )

        # 模型参数
        params = {
            'objective': 'multiclass',
            'num_class': len(y_train.unique()),
            'learning_rate': config["learning_rate"],
            'max_depth': config["max_depth"],
            'num_leaves': 31,  # 添加 num_leaves 参数
            'metric': 'multi_logloss',
            'seed': 42
        }

        # 训练模型
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200
        )
        return model
    except Exception as e:
        print(f"训练模型时发生错误：{e}")
        raise e


def evaluate_model(model, X_test, y_test, label_encoder):
    """评估模型在测试集上的性能"""
    try:
        # 预测
        y_prob = model.predict(X_test)
        y_pred = y_prob.argmax(axis=1)

        # 获取类别名称
        class_names = label_encoder.classes_

        # 计算各项指标
        accuracy = accuracy_score(y_test, y_pred)
        precisions = precision_score(y_test, y_pred, average=None)

        print("\n评估结果：")
        print(f"整体准确率：{accuracy:.4f}")
        print("\n各类别精确度：")
        for i, name in enumerate(class_names):
            print(f"{name}: {precisions[i]:.4f}")

        # 打印完整分类报告（可选）
        print("\n详细分类报告：")
        print(classification_report(y_test, y_pred, target_names=class_names))

    except Exception as e:
        print(f"评估模型时发生错误：{e}")
        raise e


def train_model_and_save(config, data_path, weight_path, model_path, label_encoder_path, feature_columns_path):
    """训练 LightGBM 模型并保存（新增测试集评估）"""
    try:
        # 数据预处理
        X, y, label_encoder, sample_weights = load_and_preprocess_data(data_path, weight_path)

        # 保存特征列
        joblib.dump(X.columns.tolist(), feature_columns_path)

        # 划分数据集为训练集和测试集
        X_train, X_test, y_train, y_test, weights_train, _ = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )

        # 训练模型
        model = train_lightgbm_model(X_train, y_train, weights_train, config)

        # 评估模型
        evaluate_model(model, X_test, y_test, label_encoder)

        # 保存模型和预处理对象
        model.save_model(model_path)
        joblib.dump(label_encoder, label_encoder_path)
        print("模型训练完成并已保存，专家权重已成功应用！")

    except Exception as e:
        print(f"训练模型并保存时发生错误：{e}")
        raise e


# if __name__ == "__main__":
#     # 模型参数
#     config = {
#         "n_estimators": 200,
#         "learning_rate": 0.1,
#         "max_depth": 6,
#         "random_state": 42
#     }
#
#     # 路径配置
#     data_path = "../Data/timeseries_data.csv"
#     weight_path = "../Data/point_id_weight.csv"
#     model_path = "../Model/lightGBM_model/lightgbm_model.txt"
#     label_encoder_path = "../Model/lightGBM_model/label_encoder.pkl"
#     feature_columns_path = "../Model/lightGBM_model/feature_columns.pkl"
#
#     # 训练模型并保存
#     train_model_and_save(
#         config,
#         data_path,
#         weight_path,
#         model_path,
#         label_encoder_path,
#         feature_columns_path
#     )