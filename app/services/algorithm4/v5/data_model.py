# 数据模型模块
import json
import numpy as np
import time
from typing import Dict, List, Any, Tuple
from config import RESOURCE_TYPES, RESOURCE_SUBTYPES, WORKSHOPS

class ResourceData:
    """资源数据类，用于处理和存储资源分配数据"""
    
    def __init__(self, data: Dict[str, Any] = None):
        """初始化资源数据
        
        Args:
            data: 资源数据字典，如果为None则创建空数据结构
        """
        if data:
            self.data = data
        else:
            # 创建空数据结构
            self.data = {}
            for resource_type in RESOURCE_TYPES:
                self.data[resource_type] = {
                    "title": self._get_title(resource_type),
                    "data": []
                }
                for workshop in WORKSHOPS:
                    self.data[resource_type]["data"].append({
                        "name": workshop,
                        "value": 0,
                        "color": "#CCCCCC"
                    })
                
                # 添加子类型数据结构
                if resource_type in RESOURCE_SUBTYPES and RESOURCE_SUBTYPES[resource_type]:
                    self.data[resource_type]["subtypes"] = {}
                    for subtype in RESOURCE_SUBTYPES[resource_type]:
                        subtype_id = subtype["id"]
                        self.data[resource_type]["subtypes"][subtype_id] = {
                            "title": subtype["name"],
                            "data": []
                        }
                        for workshop in WORKSHOPS:
                            self.data[resource_type]["subtypes"][subtype_id]["data"].append({
                                "name": workshop,
                                "value": 0,
                                "color": "#CCCCCC"
                            })
        
        # 性能指标
        self.performance_metrics = {
            "response_time": 0,  # 响应耗时（毫秒）
            "response_timeliness": 0,  # 响应时效（0-1）
            "response_quality": 0,  # 响应质量（0-1）
            "resource_utilization": 0,  # 资源利用率（0-1）
            "event_completion_rate": 0  # 事件完成率（0-1）
        }
    
    def _get_title(self, resource_type: str) -> str:
        """获取资源类型的标题
        
        Args:
            resource_type: 资源类型
            
        Returns:
            资源类型的中文标题
        """
        titles = {
            "personnel": "人员分配",
            "materials": "物料分配",
            "electricity": "电力分配"
        }
        return titles.get(resource_type, resource_type)
    
    def load_from_file(self, file_path: str) -> None:
        """从文件加载资源数据
        
        Args:
            file_path: 文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"加载资源数据失败: {e}")
    
    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """从字典加载资源数据
        
        Args:
            data: 资源数据字典
        """
        self.data = data
    
    def save_to_file(self, file_path: str) -> None:
        """保存资源数据到文件
        
        Args:
            file_path: 文件路径
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存资源数据失败: {e}")
    
    def get_resource_matrix(self) -> Dict[str, np.ndarray]:
        """获取资源分配矩阵
        
        Returns:
            资源分配矩阵字典，键为资源类型，值为numpy数组
        """
        resource_matrix = {}
        
        # 处理主资源类型
        for resource_type in RESOURCE_TYPES:
            if resource_type in self.data:
                values = [item["value"] for item in self.data[resource_type]["data"]]
                resource_matrix[resource_type] = np.array(values)
        
        # 处理子资源类型
        for resource_type in RESOURCE_TYPES:
            if resource_type in self.data and "subtypes" in self.data[resource_type]:
                for subtype_id, subtype_data in self.data[resource_type]["subtypes"].items():
                    subtype_key = f"{resource_type}_{subtype_id}"
                    values = [item["value"] for item in subtype_data["data"]]
                    resource_matrix[subtype_key] = np.array(values)
                    
                    # 更新主资源类型的分配，确保主资源类型的分配是子类型的总和
                    if resource_type in resource_matrix:
                        resource_matrix[resource_type] = np.add(resource_matrix[resource_type], values)
                    else:
                        resource_matrix[resource_type] = np.array(values)
        
        return resource_matrix
    
    def update_from_matrix(self, resource_matrix: Dict[str, np.ndarray]) -> None:
        """从资源分配矩阵更新数据
        
        Args:
            resource_matrix: 资源分配矩阵字典，键为资源类型，值为numpy数组
        """
        # 首先清空主资源类型的值，以便后续从子类型累加
        for resource_type in RESOURCE_TYPES:
            if resource_type in self.data:
                for i in range(len(self.data[resource_type]["data"])):
                    self.data[resource_type]["data"][i]["value"] = 0.0
                    self.data[resource_type]["data"][i]["color"] = "#CCCCCC"
        
        # 先更新子资源类型
        for resource_type in RESOURCE_TYPES:
            if resource_type in self.data and "subtypes" in self.data[resource_type]:
                for subtype_id in self.data[resource_type]["subtypes"]:
                    subtype_key = f"{resource_type}_{subtype_id}"
                    if subtype_key in resource_matrix:
                        for i, value in enumerate(resource_matrix[subtype_key]):
                            if i < len(self.data[resource_type]["subtypes"][subtype_id]["data"]):
                                # 确保值不为负
                                value = max(0, value)
                                
                                # 如果是人员资源类型，确保值为整数
                                if resource_type == 'personnel':
                                    value = round(value)  # 四舍五入为整数
                                else:
                                    # 对其他资源类型，四舍五入到整数
                                    value = round(value)
                                    
                                self.data[resource_type]["subtypes"][subtype_id]["data"][i]["value"] = float(value)
                                # 根据值的大小设置颜色
                                self.data[resource_type]["subtypes"][subtype_id]["data"][i]["color"] = self._get_color_by_value(value)
                                
                                # 累加到主资源类型
                                self.data[resource_type]["data"][i]["value"] += float(value)
                                # 更新主资源类型的颜色
                                self.data[resource_type]["data"][i]["color"] = self._get_color_by_value(self.data[resource_type]["data"][i]["value"])
        
        # 然后更新主资源类型（如果资源矩阵中有主资源类型的数据）
        for resource_type in RESOURCE_TYPES:
            if resource_type in resource_matrix and resource_type in self.data:
                # 只有当没有子类型时，才直接使用主资源类型的数据
                if "subtypes" not in self.data[resource_type] or not self.data[resource_type]["subtypes"]:
                    for i, value in enumerate(resource_matrix[resource_type]):
                        if i < len(self.data[resource_type]["data"]):
                            # 确保值不为负
                            value = max(0, value)
                            
                            # 如果是人员资源类型，确保值为整数
                            if resource_type == 'personnel':
                                value = round(value)  # 四舍五入为整数
                            else:
                                # 对其他资源类型，四舍五入到整数
                                value = round(value)
                                
                            self.data[resource_type]["data"][i]["value"] = float(value)
                            # 根据值的大小设置颜色
                            self.data[resource_type]["data"][i]["color"] = self._get_color_by_value(value)
        
        # 子资源类型已在前面更新完成
    
    def _get_color_by_value(self, value: float) -> str:
        """根据值的大小获取颜色
        
        Args:
            value: 值
            
        Returns:
            颜色代码
        """
        # 根据值的大小返回不同的颜色
        if value < 10:
            return "#CCCCCC"  # 灰色
        elif value < 30:
            return "#66CC66"  # 浅绿色
        elif value < 60:
            return "#3399FF"  # 蓝色
        elif value < 90:
            return "#FF9933"  # 橙色
        else:
            return "#FF6666"  # 红色
    
    def get_total_resources(self) -> Dict[str, float]:
        """获取资源总量
        
        Returns:
            资源总量字典，键为资源类型，值为总量
        """
        total_resources = {}
        
        # 计算主资源类型总量
        for resource_type in RESOURCE_TYPES:
            if resource_type in self.data:
                total = sum(item["value"] for item in self.data[resource_type]["data"])
                total_resources[resource_type] = total
        
        # 计算子资源类型总量
        for resource_type in RESOURCE_TYPES:
            if resource_type in self.data and "subtypes" in self.data[resource_type]:
                for subtype_id, subtype_data in self.data[resource_type]["subtypes"].items():
                    subtype_key = f"{resource_type}_{subtype_id}"
                    total = sum(item["value"] for item in subtype_data["data"])
                    total_resources[subtype_key] = total
        
        return total_resources
    
    def calculate_performance_metrics(self, start_time: float, iterations: int, converged: bool) -> None:
        """计算性能指标
        
        Args:
            start_time: 开始时间
            iterations: 迭代次数
            converged: 是否收敛
        """
        # 计算响应时间（毫秒）
        response_time = (time.time() - start_time) * 1000
        self.performance_metrics["response_time"] = response_time
        
        # 计算执行时间（秒），用于run_example.py中显示
        execution_time = (time.time() - start_time)
        self.performance_metrics["execution_time"] = execution_time
        
        # 添加迭代次数和收敛状态到性能指标
        self.performance_metrics["iterations"] = iterations
        self.performance_metrics["converged"] = converged
        
        # 添加时间戳，用于报告生成
        self.performance_metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # 计算响应时效（基于响应时间的归一化值）
        # 根据实际响应时间计算时效性，假设理想响应时间为100ms，最大容忍时间为10000ms
        response_timeliness = max(0, min(1, 1 - (response_time - 100) / 9900))
        self.performance_metrics["response_timeliness"] = response_timeliness
        
        # 计算响应质量（基于是否收敛和迭代次数）
        if converged:
            # 如果收敛，质量基于迭代次数，迭代次数越少质量越高
            # 假设迭代次数小于100为最佳，大于1000为最差
            response_quality = max(0, min(1, 1 - (iterations - 100) / 900))
        else:
            # 如果未收敛，质量固定为较低值
            response_quality = 0.3
        self.performance_metrics["response_quality"] = response_quality
        
        # 计算资源利用率（基于资源分配的均衡性）
        resource_utilization = self._calculate_resource_utilization()
        self.performance_metrics["resource_utilization"] = resource_utilization
        
        # 计算事件完成率（基于响应质量和资源利用率）
        event_completion_rate = (response_quality + resource_utilization) / 2
        self.performance_metrics["event_completion_rate"] = event_completion_rate
    
    def _calculate_resource_utilization(self) -> float:
        """计算资源利用率
        
        Returns:
            资源利用率（0-1）
        """
        # 计算各资源类型的分配均衡性
        balance_scores = []
        
        for resource_type in RESOURCE_TYPES:
            if resource_type in self.data:
                values = [item["value"] for item in self.data[resource_type]["data"]]
                if sum(values) > 0:
                    # 计算变异系数（标准差/均值），越小越均衡
                    mean = np.mean(values)
                    std = np.std(values)
                    cv = std / mean if mean > 0 else 0
                    # 将变异系数转换为均衡性得分（0-1），变异系数越小得分越高
                    balance_score = max(0, min(1, 1 - cv))
                    balance_scores.append(balance_score)
        
        # 如果没有有效的均衡性得分，返回默认值
        if not balance_scores:
            return 0.5
        
        # 返回平均均衡性得分作为资源利用率
        return sum(balance_scores) / len(balance_scores)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标
        
        Returns:
            性能指标字典
        """
        return self.performance_metrics