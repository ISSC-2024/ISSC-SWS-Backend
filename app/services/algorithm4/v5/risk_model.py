# 风险模型模块
import numpy as np
from typing import Dict, List, Any
from config import WORKSHOPS

# 风险等级阈值
RISK_THRESHOLDS = {
    'low': 0.3,      # 低风险阈值
    'medium': 0.6,   # 中风险阈值
    'high': 0.8      # 高风险阈值
}

# 风险等级权重
RISK_LEVELS = {
    'low': 1,        # 低风险权重
    'medium': 2,     # 中风险权重
    'high': 3,       # 高风险权重
    'critical': 4    # 极高风险权重
}

class RiskModel:
    """风险模型类，用于模拟从前三个模块获取的风险评估数据"""
    
    def __init__(self):
        """初始化风险模型"""
        self.workshop_risks = {}
        self.last_update_time = None  # 上次更新时间
        self.initialize_risks()
    
    def initialize_risks(self) -> None:
        """初始化风险数据，从CSV文件读取前序模块输出的风险数据，只考虑风险等级"""
        import pandas as pd
        import os
        from config import RISK_DATA_FILE_PATH, WORKSHOPS
        
        # 初始化风险数据结构
        for workshop in WORKSHOPS:
            self.workshop_risks[workshop] = {
                'risk_score': 0.0,  # 风险得分，范围[0, 1]
                'risk_factors': {
                    'equipment': 0.0,  # 设备风险
                    'process': 0.0,  # 工艺风险
                    'environment': 0.0,  # 环境风险
                    'human': 0.0  # 人员风险
                },
                'historical_incidents': 0  # 历史事故数量
            }
        
        # 检查风险数据文件是否存在
        if not os.path.exists(RISK_DATA_FILE_PATH):
            print(f"警告: 风险数据文件 {RISK_DATA_FILE_PATH} 不存在，使用随机数据初始化")
            self._initialize_random_risks()
            return
            
        try:
            # 读取CSV文件
            risk_data = pd.read_csv(RISK_DATA_FILE_PATH)
            
            # 处理数据，按车间区域分组并计算风险得分
            # 从point_id提取车间代码（前三个字符）
            risk_data['workshop_code'] = risk_data['point_id'].str[:3]
            
            # 创建车间代码到车间名称的映射
            workshop_code_map = {
                'RMS': '原料储存区',
                'REA': '反应器区',
                'SEP': '分离提纯区',
                'PRO': '成品储存区',
                'UTL': '公用工程区'
            }
            
            # 将风险级别映射为数值
            risk_level_map = {
                'safe': 0.1,  # 安全
                'warning': 0.6,  # 警告
                'danger': 0.8,  # 危险
                'critical': 1.0  # 严重
            }
            
            # 将风险级别转换为数值
            risk_data['risk_score'] = risk_data['risk_level'].map(risk_level_map)
            
            # 按车间分组并计算平均风险得分
            workshop_risks = risk_data.groupby('workshop_code').agg({
                'risk_score': 'mean'
            })
            
            # 计算各车间的警告数量作为历史事故数量
            warning_counts = risk_data[risk_data['risk_level'] != 'safe'].groupby('workshop_code').size()
            
            # 更新风险模型数据
            for code, name in workshop_code_map.items():
                if code in workshop_risks.index:
                    data = workshop_risks.loc[code]
                    
                    # 设置默认的风险因素得分（所有因素使用相同的风险得分）
                    risk_score = data['risk_score']
                    
                    # 更新风险数据
                    self.workshop_risks[name] = {
                        'risk_score': risk_score,
                        'risk_factors': {
                            'equipment': risk_score,  # 所有风险因素使用相同的风险得分
                            'process': risk_score,
                            'environment': risk_score,
                            'human': risk_score
                        },
                        'historical_incidents': warning_counts.get(code, 0)  # 历史事故数量
                    }
        except Exception as e:
            print(f"读取风险数据文件出错: {e}，使用随机数据初始化")
            self._initialize_random_risks()
    
    def _initialize_random_risks(self) -> None:
        """使用随机数据初始化风险模型（作为备用方案）"""
        for workshop in WORKSHOPS:
            self.workshop_risks[workshop] = {
                'risk_score': np.random.uniform(0, 1),  # 风险得分，范围[0, 1]
                'risk_factors': {
                    'equipment': np.random.uniform(0, 1),  # 设备风险
                    'process': np.random.uniform(0, 1),  # 工艺风险
                    'environment': np.random.uniform(0, 1),  # 环境风险
                    'human': np.random.uniform(0, 1)  # 人员风险
                },
                'historical_incidents': int(np.random.poisson(2))  # 历史事故数量
            }
    
    def update_risks(self, new_data: Dict[str, Any] = None) -> None:
        """更新风险数据，每5秒更新一次
        
        Args:
            new_data: 新的风险数据，如果为None则从CSV文件重新读取
        """
        import time
        current_time = time.time()
        
        # 检查是否需要更新（首次更新或者距离上次更新已经过了5秒）
        if self.last_update_time is None or (current_time - self.last_update_time) >= 5.0:
            if new_data:
                # 使用提供的数据更新风险
                for workshop, risk_data in new_data.items():
                    if workshop in self.workshop_risks:
                        self.workshop_risks[workshop].update(risk_data)
            else:
                # 从CSV文件重新读取风险数据
                try:
                    # 重新初始化风险数据
                    self.initialize_risks()
                except Exception as e:
                    print(f"更新风险数据出错: {e}，使用随机波动更新")
                    # 如果读取失败，则使用随机波动更新
                    self._update_risks_with_random_fluctuation()
            
            # 更新上次更新时间
            self.last_update_time = current_time
            print(f"风险数据已更新，时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}")
        else:
            # 距离上次更新不足5秒，跳过更新
            remaining = 5.0 - (current_time - self.last_update_time)
            print(f"距离上次更新不足5秒，跳过更新。还需等待 {remaining:.2f} 秒")
    
    def _update_risks_with_random_fluctuation(self) -> None:
        """使用随机波动更新风险数据（作为备用方案）"""
        for workshop in WORKSHOPS:
            # 在原有基础上小幅度波动
            current_risk = self.workshop_risks[workshop]['risk_score']
            # 添加随机波动，但保持在[0, 1]范围内
            new_risk = np.clip(current_risk + np.random.uniform(-0.1, 0.1), 0, 1)
            self.workshop_risks[workshop]['risk_score'] = new_risk
            
            # 更新风险因素
            for factor in self.workshop_risks[workshop]['risk_factors']:
                current_factor = self.workshop_risks[workshop]['risk_factors'][factor]
                new_factor = np.clip(current_factor + np.random.uniform(-0.05, 0.05), 0, 1)
                self.workshop_risks[workshop]['risk_factors'][factor] = new_factor
    
    def get_risk_levels(self) -> Dict[str, str]:
        """获取各车间的风险等级
        
        Returns:
            风险等级字典，键为车间名称，值为风险等级
        """
        risk_levels = {}
        for workshop, risk_data in self.workshop_risks.items():
            risk_score = risk_data['risk_score']
            
            if risk_score < RISK_THRESHOLDS['low']:
                level = 'low'
            elif risk_score < RISK_THRESHOLDS['medium']:
                level = 'medium'
            elif risk_score < RISK_THRESHOLDS['high']:
                level = 'high'
            else:
                level = 'critical'
                
            risk_levels[workshop] = level
        
        return risk_levels
    
    def get_risk_priority_weights(self) -> Dict[str, float]:
        """获取基于风险的优先级权重
        
        Returns:
            优先级权重字典，键为车间名称，值为权重
        """
        weights = {}
        risk_levels = self.get_risk_levels()
        
        # 计算总风险值作为归一化因子
        total_risk = sum(RISK_LEVELS[level] for level in risk_levels.values())
        
        if total_risk > 0:
            for workshop, level in risk_levels.items():
                # 风险等级越高，权重越大
                weights[workshop] = RISK_LEVELS[level] / total_risk
        else:
            # 如果总风险为0，则平均分配
            equal_weight = 1.0 / len(WORKSHOPS)
            weights = {workshop: equal_weight for workshop in WORKSHOPS}
        
        return weights
    
    def get_risk_data(self) -> Dict[str, Any]:
        """获取风险数据
        
        Returns:
            风险数据字典
        """
        return {
            'workshop_risks': self.workshop_risks,
            'risk_levels': self.get_risk_levels(),
            'risk_priority_weights': self.get_risk_priority_weights()
        }