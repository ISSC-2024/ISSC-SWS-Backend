import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb  # 修改导入方式


def load_expert_weights(weight_path):
    """加载专家权重文件（保持不变）"""
    try:
        weight_df = pd.read_csv(weight_path)
        return dict(zip(weight_df['point_id'], weight_df['weight']))
    except Exception as e:
        print(f"加载专家权重文件时发生错误：{e}")
        raise e


def generate_feature_weights(feature_columns, expert_weights):
    """生成特征权重数组（保持不变）"""
    try:
        weights = []
        for col in feature_columns:
            if col.startswith('point_id_'):
                raw_id = col.split('_', 2)[-1]
                weights.append(expert_weights.get(raw_id, 1.0))
            else:
                weights.append(1.0)
        return weights
    except Exception as e:
        print(f"生成特征权重时发生错误：{e}")
        raise e


def load_and_preprocess_data(data_path, weight_path):
    """加载数据并进行预处理（保持不变）"""
    try:
        expert_weights = load_expert_weights(weight_path)
        df = pd.read_csv(data_path)
        df.drop('timestamp', axis=1, inplace=True)

        label_encoder = LabelEncoder()
        df['risk_level'] = label_encoder.fit_transform(df['risk_level'])

        df = pd.get_dummies(df, columns=['gas_type', 'point_id'])
        feature_weights = generate_feature_weights(df.columns.tolist(), expert_weights)

        return df, label_encoder, feature_weights
    except Exception as e:
        print(f"加载和预处理数据时发生错误：{e}")
        raise e


def train_xgboost_model(X_train, y_train, config, feature_weights):
    """训练XGBoost模型（使用原生接口）"""
    try:
        # 转换为DMatrix并传入特征权重
        dtrain = xgb.DMatrix(
            X_train,
            label=y_train,
            feature_weights=feature_weights  # 关键修改点
        )

        params = {
            'objective': 'multi:softprob',
            'num_class': y_train.nunique(),
            'learning_rate': config["learning_rate"],
            'max_depth': config["max_depth"],
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
            'eval_metric': 'mlogloss'
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200
        )
        return model
    except Exception as e:
        print(f"训练模型时发生错误：{e}")
        raise e


def train_model_and_save(config, data_path, weight_path, model_path, label_encoder_path, feature_columns_path):
    """训练模型并保存（新增测试集评估）"""
    try:
        df, label_encoder, feature_weights = load_and_preprocess_data(data_path, weight_path)
        X = df.drop(['risk_level', 'risk_level_name'], axis=1)
        y = df['risk_level']

        # 修改分割方式，保留测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y  # 建议保持类别分布
        )

        # 训练模型
        model = train_xgboost_model(X_train, y_train, config, feature_weights)

        # 新增评估部分
        # 修改后的评估部分
        dtest = xgb.DMatrix(X_test)
        y_prob = model.predict(dtest)
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

        # 保存模型和元数据
        model.save_model(model_path)
        joblib.dump(label_encoder, label_encoder_path)
        joblib.dump(X.columns.tolist(), feature_columns_path)

        print("模型训练完成并已保存，特征权重已应用！")
    except Exception as e:
        print(f"训练模型并保存时发生错误：{e}")
        raise e

# def train_model_and_save(config, data_path, weight_path, model_path, label_encoder_path, feature_columns_path):
#     """训练模型并保存（使用全部数据）"""
#     try:
#         df, label_encoder, feature_weights = load_and_preprocess_data(data_path, weight_path)
#         X = df.drop(['risk_level', 'risk_level_name'], axis=1)
#         y = df['risk_level']
#
#         # 使用全部数据进行训练
#         model = train_xgboost_model(X, y, config, feature_weights)
#
#         # 保存模型和元数据
#         model.save_model(model_path)
#         joblib.dump(label_encoder, label_encoder_path)
#         joblib.dump(X.columns.tolist(), feature_columns_path)
#
#         print("模型训练完成并已保存，特征权重已应用！")
#     except Exception as e:
#         print(f"训练模型并保存时发生错误：{e}")
#         raise e

# if __name__ == "__main__":
#     config = {
#         "n_estimators": 200,
#         "learning_rate": 0.1,
#         "max_depth": 6,
#         "random_state": 42
#     }
#     train_model_and_save(
#         config,
#         "../Data/timeseries_data.csv",
#         "../Data/point_id_weight.csv",
#         "../Model/xgboost_model/xgboost_model.ubj",  # 已修改扩展名
#         "../Model/xgboost_model/label_encoder.pkl",
#         "../Model/xgboost_model/feature_columns.pkl"
#     )