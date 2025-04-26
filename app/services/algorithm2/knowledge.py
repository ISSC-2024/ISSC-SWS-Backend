import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class LightweightKnowledgeGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_index = {}
    
    def add_node(self, node_id, label, **properties):
        if node_id not in self.node_index:
            node = {
                "id": node_id,
                "label": label,
                "properties": properties
            }
            self.nodes.append(node)
            self.node_index[node_id] = len(self.nodes)-1
        return self.node_index[node_id]
    
    def add_edge(self, source_id, target_id, rel_type):
        edge = {
            "source": source_id,
            "target": target_id,
            "type": rel_type
        }
        self.edges.append(edge)
    
    def build_graph(self, points_data, risk_predictions, metadata):
        # 创建区域节点
        area_nodes = {}
        for area in metadata['areas']:
            node_id = f"Area_{area['code']}"
            self.add_node(node_id, "Area", 
                        code=area['code'],
                        name=area['name'],
                        description=area['description'])
            area_nodes[area['code']] = node_id
        
        # 创建风险等级节点
        risk_nodes = {
            '安全': self.add_node("Risk_safe", "RiskLevel", 
                                level="安全", severity=0),
            '警告': self.add_node("Risk_warning", "RiskLevel", 
                                 level="警告", severity=0.5),
            '危险': self.add_node("Risk_danger", "RiskLevel", 
                                  level="危险", severity=1)
        }
        
        # 创建监测点及关系
        sensor_cache = {}
        for _, row in points_data.iterrows():
            point_id = row['point_id']
            weight = row['weight']  # 从合并后的数据列获取
            
            # 创建监测点节点
            self.add_node(
                point_id, "MonitoringPoint",
                x=row['x_coordinate'],
                y=row['y_coordinate'],
                description=row['description'],
                weight=weight
            )
            
            # 关联区域
            self.add_edge(point_id, area_nodes[row['area_code']], "BELONGS_TO")
            
            # 关联传感器
            for sensor in row['installed_sensors'].split(','):
                sensor = sensor.strip()
                if sensor not in sensor_cache:
                    sensor_id = f"Sensor_{sensor}"
                    self.add_node(sensor_id, "Sensor",
                                type=sensor,
                                unit=metadata['sensors'][sensor]['unit'])
                    sensor_cache[sensor] = sensor_id
                self.add_edge(point_id, sensor_cache[sensor], "HAS_SENSOR")
            
        # 添加风险关联逻辑
        for point_id, risk_label in risk_predictions.items():
            if risk_label in risk_nodes:
                self.add_edge(point_id, risk_nodes[risk_label], "HAS_RISK")
            else:
                print(f"无效风险标签: {point_id}->{risk_label} (有效值：安全/警告/危险)")
        
    def visualize(self):
        try:
            G = nx.Graph()
            
            # 添加节点（修复属性结构）
            for node in self.nodes:
                node_attrs = {"label": node["label"]}
                node_attrs.update(node["properties"])
                G.add_node(node["id"], **node_attrs)
            
            # 添加边
            for edge in self.edges:
                G.add_edge(edge["source"], edge["target"], 
                          label=edge["type"])
            
            # 可视化配置
            plt.figure(figsize=(24, 18))
            pos = nx.spring_layout(G, k=0.6, iterations=50)
            
            # 节点颜色映射
            node_colors = {
                "Area": "#FFD700",
                "MonitoringPoint": "#87CEEB", 
                "Sensor": "#98FB98",
                "RiskLevel": "#FF6347"
            }
            
            # 绘制节点
            for node_type, color in node_colors.items():
                nodes = [
                    n for n in G.nodes 
                    if G.nodes[n].get("label") == node_type
                ]
                nx.draw_networkx_nodes(
                    G, pos, nodelist=nodes,
                    node_color=color,
                    node_size=1200,
                    alpha=0.9
                )
            
            # 绘制边
            nx.draw_networkx_edges(
                G, pos, 
                edge_color="gray", 
                width=0.8,
                alpha=0.4
            )
            
            # 生成标签
            labels = {
                n: "\n".join([
                    G.nodes[n].get("label", "Node"),
                    *[f"{k}: {v}" for k, v in G.nodes[n].items() 
                     if k not in ["label"] and v]
                ])
                for n in G.nodes
            }
            
            # 绘制标签
            nx.draw_networkx_labels(
                G, pos, labels,
                font_size=8,
                font_family="SimHei",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7
                )
            )
            
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("knowledge_graph.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("可视化图表已保存为 knowledge_graph.png")
            
        except Exception as e:
            print(f"可视化错误: {str(e)}")
            raise

    def save_graph(self, filename="knowledge_graph.json"):
        """保存图谱到JSON文件"""
        graph_data = {
            "nodes": self.nodes,
            "edges": self.edges
        }
        with open(filename, 'w',encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

def main():
    # 数据加载
    ts_data = pd.read_csv(r"chemical_plant_dataset_1day\timeseries_data.csv", parse_dates=['timestamp'])
    points_data = pd.read_csv(r"chemical_plant_dataset_1day\monitoring_points.csv")
    
    with open(r"chemical_plant_dataset_1day\metadata.json",encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 特征工程
    ts_data['area_code'] = ts_data['point_id'].str[:3]

    # 定义偏离度计算函数
    def calculate_deviation(row):
        sensor_type = None
        value = None
        area = row['area_code']
        
        # 判断当前处理的传感器类型
        if 'temperature' in row:
            sensor_type = 'temperature'
            value = row[sensor_type]
        elif 'pressure' in row:
            sensor_type = 'pressure'
            value = row[sensor_type]
        elif 'flow_rate' in row:
            sensor_type = 'flow_rate'
            value = row[sensor_type]
        elif 'level' in row:
            sensor_type = 'level'
            value = row[sensor_type]
        else:  # 气体浓度特殊处理
            sensor_type = 'gas_concentration'
            gas_type = row['gas_type']
            value = row[sensor_type]
            normal_range = metadata['sensors'][sensor_type]['normal_ranges'][gas_type]
            lower, upper = normal_range
            deviation = max(lower - value, value - upper, 0)
            return deviation / (upper - lower) if deviation > 0 else 0
        
        # 常规传感器处理
        normal_range = metadata['sensors'][sensor_type]['normal_ranges'][area]
        lower, upper = normal_range
        deviation = max(lower - value, value - upper, 0)
        return deviation / (upper - lower) if deviation > 0 else 0

    # 特征生成
    features_list = []
    window_size = 60  # 60个时间点（对应10分钟窗口）
    
    for point_id, group in tqdm(ts_data.groupby('point_id'), desc="Processing Points"):
        #! 按时间排序（暂时禁用）
        # group = group.sort_values('timestamp')
        
        # 基础特征
        for col in ['temperature', 'pressure', 'flow_rate', 'level', 'gas_concentration']:
            # 滚动统计特征
            group[f'{col}_mean'] = group[col].rolling(window_size, min_periods=1).mean()
            group[f'{col}_std'] = group[col].rolling(window_size, min_periods=1).std()
            
            # 变化率特征
            group[f'{col}_delta'] = group[col].diff().fillna(0)
        
        # 偏离度特征
        group['temp_dev'] = group.apply(
            lambda x: calculate_deviation(x[['temperature', 'area_code']]), axis=1)
        group['pressure_dev'] = group.apply(
            lambda x: calculate_deviation(x[['pressure', 'area_code']]), axis=1)
        group['flow_dev'] = group.apply(
            lambda x: calculate_deviation(x[['flow_rate', 'area_code']]), axis=1)
        group['level_dev'] = group.apply(
            lambda x: calculate_deviation(x[['level', 'area_code']]), axis=1)
        group['gas_dev'] = group.apply(
            lambda x: calculate_deviation(x[['gas_concentration', 'gas_type', 'area_code']]), axis=1)
        
        # 交互特征
        group['temp_pressure_ratio'] = group['temperature'] / (group['pressure'] + 1e-6)
        group['flow_level_ratio'] = group['flow_rate'] / (group['level'] + 1e-6)
        
        # 特征标记
        group['is_peak_hour'] = group['timestamp'].dt.hour.isin([8, 12, 18]).astype(int)
        
        features_list.append(group)

    features = pd.concat(features_list)
    
     # 模型训练
    le = LabelEncoder()
    features['risk_level'] = le.fit_transform(features['risk_level'])
    
    X = features[['temp_dev', 'pressure_dev', 'gas_dev',
                 'temperature_mean', 'pressure_std', 'gas_concentration_std']]
    y = features['risk_level']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = RandomForestClassifier(n_estimators=150, 
                                  class_weight='balanced',
                                  max_depth=6,
                                  random_state=42)
    model.fit(X_train, y_train)
    
    # 预测概率
    latest_data = features.groupby('point_id').last().reset_index()
    proba = model.predict_proba(
        latest_data[['temp_dev', 'pressure_dev', 'gas_dev',
                   'temperature_mean', 'pressure_std', 'gas_concentration_std']])
    
    # 获取类别顺序
    class_order = le.classes_.tolist()
    weight_map = {'safe': 0.0, 'warning': 0.5, 'danger': 1.0}
    
    # 动态映射类别
    weights = []
    for p in proba:
        weight = 0.0
        for idx, cls in enumerate(class_order):
            weight += p[idx] * weight_map.get(cls, 0.0)
        weights.append(np.clip(weight, 0.0, 1.0))
    
    # 构建结果
    output_df = latest_data[['point_id']].merge(
        points_data[['point_id', 'area_code', 'area_name']],
        on='point_id'
    )
    X_latest = latest_data[['temp_dev', 'pressure_dev', 'gas_dev',
                      'temperature_mean', 'pressure_std', 'gas_concentration_std']]
    output_df['pred_risk'] = le.inverse_transform(model.predict(X_latest))

    output_df['weight'] = weights
    
    # 保存结果
    output_df.to_csv("monitoring_points_weights.csv", index=False)

    # 将权重合并到points_data
    points_data = points_data.merge(
        output_df[['point_id', 'weight']], 
        on='point_id', 
        how='left'
    )
    
    # 构建知识图谱
    risk_predictions = dict(zip(output_df['point_id'], output_df['pred_risk']))
    lkg = LightweightKnowledgeGraph()
    lkg.build_graph(points_data, risk_predictions, metadata)
    lkg.visualize()
    lkg.save_graph("knowledge_graph.json")

if __name__ == "__main__":
    main()