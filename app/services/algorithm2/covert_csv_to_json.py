import csv
import json
import os
import math
import itertools

# 读取CSV文件
def read_csv(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

# 生成知识图谱JSON数据
def generate_graph_json(csv_data):
    nodes = []
    links = []
    categories = [
        {"name": "区域"},
        {"name": "传感器"},
        {"name": "安全"},
        {"name": "警告"},
        {"name": "危险"}
    ]
    
    # 添加区域节点
    area_codes = {}
    for row in csv_data:
        if row['area_code'] not in area_codes:
            area_codes[row['area_code']] = row['area_name']
    
    area_nodes = {}
    for code, name in area_codes.items():
        # 区域节点ID使用区域代码
        area_nodes[code] = code
        nodes.append({
            "id": code,
            "name": name,
            "symbolSize": 80,
            "category": 0,
            "itemStyle": {"color": "#5470c6"}  # 区域节点使用蓝色
        })
    
    # 连接所有区域节点 - 形成区域网络
    for area1, area2 in itertools.combinations(area_codes.keys(), 2):
        links.append({
            "source": area1,
            "target": area2,
            "value": 2  # 区域间连接较细
        })
    
    # 添加传感器节点和连接
    sensor_nodes = {}
    for row in csv_data:
        sensor_id = row['point_id']
        area_code = row['area_code']
        
        # 传感器节点ID使用区域代码+传感器ID组合
        node_id = f"{area_code}_{sensor_id}"
        
        if node_id not in sensor_nodes:
            sensor_nodes[sensor_id] = node_id
            
            # 添加传感器节点
            nodes.append({
                "id": node_id,
                "name": sensor_id,
                "symbolSize": 40,
                "category": 1,
                "weight": float(row['weight']),
                "area_code": area_code,
                "pred_risk": row['pred_risk'],
                "itemStyle": {"color": "#91cc75"}  # 传感器节点使用绿色
            })
            
            # 连接区域到传感器
            links.append({
                "source": area_code,  # 区域节点ID
                "target": node_id,    # 传感器节点ID
                "value": 3
            })
    
    # 为每个传感器添加3个安全状态节点
    for row in csv_data:
        sensor_id = row['point_id']
        area_code = row['area_code']
        sensor_node_id = f"{area_code}_{sensor_id}"
        weight = float(row['weight'])
        
        # 为每个传感器创建3个独立的安全状态节点
        safe_node_id = f"{sensor_node_id}_safe"
        warning_node_id = f"{sensor_node_id}_warning"
        danger_node_id = f"{sensor_node_id}_danger"
        
        # 计算节点大小 - 权重越高，危险节点越大；权重越低，安全节点越大
        safe_size = 15 + int((1 - weight) * 35)     # 权重低时安全节点大
        warning_size = 15 + int(6 * (0.5 - abs(weight - 0.5)))  # 权重接近0.5时警告节点大
        danger_size = 15 + int(weight * 35)         # 权重高时危险节点大
        
        # 添加安全节点
        nodes.append({
            "id": safe_node_id,
            "name": "安全",
            "symbolSize": safe_size,
            "category": 2,
            "itemStyle": {"color": "#67C23A"},  # 安全节点-绿色
            "sensor_id": sensor_id,
            "weight": weight
        })
        
        # 添加警告节点
        nodes.append({
            "id": warning_node_id,
            "name": "警告",
            "symbolSize": warning_size,
            "category": 3,
            "itemStyle": {"color": "#E6A23C"},  # 警告节点-黄色
            "sensor_id": sensor_id,
            "weight": weight
        })
        
        # 添加危险节点
        nodes.append({
            "id": danger_node_id,
            "name": "危险",
            "symbolSize": danger_size,
            "category": 4,
            "itemStyle": {"color": "#F56C6C"},  # 危险节点-红色
            "sensor_id": sensor_id,
            "weight": weight
        })
        
        # 连接传感器到各安全状态节点
        links.append({
            "source": sensor_node_id,
            "target": safe_node_id,
            "value": max(1, int((1 - weight) * 8))
        })
        
        links.append({
            "source": sensor_node_id,
            "target": warning_node_id,
            "value": max(1, int(6 * (0.5 - abs(weight - 0.5))))
        })
        
        links.append({
            "source": sensor_node_id,
            "target": danger_node_id,
            "value": max(1, int(weight * 8))
        })
    
    return {
        "nodes": nodes,
        "links": links,
        "categories": categories
    }

# 主函数
def main():
    input_file = 'monitoring_points_weights.csv'
    output_file = 'monitoringKnowledgeGraph.json'
    
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 读取CSV数据
    csv_data = read_csv(os.path.join(current_dir, input_file))
    
    # 生成图谱JSON
    graph_data = generate_graph_json(csv_data)
    
    # 保存JSON文件
    with open(os.path.join(current_dir, output_file), 'w', encoding='utf-8') as file:
        json.dump(graph_data, file, ensure_ascii=False, indent=2)
    
    print(f"已成功生成知识图谱数据: {output_file}")

if __name__ == "__main__":
    main()