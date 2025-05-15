# 主系统模块
import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from data_model import ResourceData
from risk_model import RiskModel
from optimization_model import ResourceOptimizer
from config import WORKSHOPS, RESOURCE_TYPES, RESOURCE_SUBTYPES, ALGORITHM_TYPES, DEFAULT_ALGORITHM_TYPE, INPUT_FILE_PATH, AGENT_PARAMS # 导入 AGENT_PARAMS

class ResourceOptimizationSystem:
    """资源优化系统类，整合数据模型、风险模型和优化模型"""
    
    def __init__(self, initial_data: Dict[str, Any] = None, algorithm_type: int = DEFAULT_ALGORITHM_TYPE):
        """初始化资源优化系统
        
        Args:
            initial_data: 初始资源数据，如果为None则使用默认数据
            algorithm_type: 算法类型，1=Independent Q-Learning, 2=DQN, 3=MADDPG, 4=MAPPO
        """
        # 初始化数据模型
        if initial_data is None:
            try:
                with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
                    initial_data = json.load(f)
            except FileNotFoundError:
                initial_data = {}
                
        self.resource_data = ResourceData(initial_data)
        
        # 初始化风险模型
        self.risk_model = RiskModel()
        
        # 初始化优化模型
        initial_resources = self.resource_data.get_resource_matrix() if initial_data else None
        self.optimizer = ResourceOptimizer(initial_resources, algorithm_type)
        
        # 优化结果
        self.optimized_resources = None
        
        # 创建数据目录
        os.makedirs('data', exist_ok=True)
        
        # 记录使用的算法类型
        self.algorithm_type = algorithm_type
        print(f"使用算法: {ALGORITHM_TYPES.get(algorithm_type, '未知算法')}")
    
    def load_data(self, file_path: str = None, data: Dict[str, Any] = None) -> None:
        """加载资源数据
        
        Args:
            file_path: 文件路径，如果提供则从文件加载
            data: 资源数据字典，如果提供则从字典加载
        """
        if file_path:
            self.resource_data.load_from_file(file_path)
        elif data:
            self.resource_data.load_from_dict(data)
    
    def save_data(self, file_path: str) -> None:
        """保存资源数据到文件
        
        Args:
            file_path: 文件路径
        """
        self.resource_data.save_to_file(file_path)
    
    def update_risk_data(self, new_risk_data: Dict[str, Any] = None) -> None:
        """更新风险数据
        
        Args:
            new_risk_data: 新的风险数据，如果为None则随机更新
        """
        self.risk_model.update_risks(new_risk_data)
    
    def optimize_resources(self, max_iterations: int = 1000) -> Dict[str, Any]:
        """优化资源分配
        
        Args:
            max_iterations: 最大迭代次数
            
        Returns:
            优化结果字典
        """
        # 获取风险权重
        risk_weights = self.risk_model.get_risk_priority_weights()
        
        # 获取资源总量
        total_resources = self.resource_data.get_total_resources()
        
        # 执行优化
        self.optimized_resources = self.optimizer.optimize(
            risk_weights, total_resources, max_iterations
        )
        
        # 更新资源数据
        self.resource_data.update_from_matrix(self.optimized_resources)
        
        # 计算性能指标
        start_time, iterations, converged = self.optimizer.get_performance_data()
        self.resource_data.calculate_performance_metrics(start_time, iterations, converged)
        
        # 获取资源流动数据
        resource_flows = self.optimizer.get_resource_flows()
        
        return {
            'optimized_resources': self.optimized_resources,
            'risk_data': self.risk_model.get_risk_data(),
            'resource_data': self.resource_data.data,
            'performance_metrics': self.resource_data.get_performance_metrics(),
            'algorithm_type': self.algorithm_type,
            'algorithm_name': ALGORITHM_TYPES.get(self.algorithm_type, '未知算法'),
            'resource_flows': resource_flows,  # 添加资源流动数据
            'resource_flow_summary': self.optimizer.get_resource_flow_summary()  # 添加资源流动总量统计
        }
    
    def get_optimization_history(self) -> List[Dict[str, np.ndarray]]:
        """获取优化历史
        
        Returns:
            优化历史列表
        """
        return self.optimizer.get_optimization_history()
    

    
    def generate_report(self, output_dir: str = 'data') -> str:
        """生成优化报告
        
        Args:
            output_dir: 输出目录路径
            
        Returns:
            报告文件路径
        """
        # 确保已经执行过优化
        if self.optimized_resources is None:
            self.optimize_resources()
        
        # 获取算法名称和收敛阈值
        algorithm_name = ALGORITHM_TYPES.get(self.algorithm_type, 'UnknownAlgorithm')
        convergence_threshold = AGENT_PARAMS.get('convergence_threshold', 0.001)
        
        # 构建文件名
        output_filename = f"{algorithm_name}__{convergence_threshold}_allocate_result.json"
        output_file = os.path.join(output_dir, output_filename)

        # 获取性能指标
        performance_metrics = self.resource_data.get_performance_metrics()
        
        # 构建报告数据
        report_data = {
            'timestamp': performance_metrics['timestamp'],
            'algorithm': {
                'type': self.algorithm_type,
                'name': algorithm_name
            },
            'performance': {
                'execution_time': performance_metrics['execution_time'],
                'iterations': performance_metrics['iterations'],
                'converged': performance_metrics['converged'],
                'response_time': performance_metrics['response_time'],
                'response_timeliness': performance_metrics['response_timeliness'],
                'response_quality': performance_metrics['response_quality'],
                'resource_utilization': performance_metrics['resource_utilization'],
                'event_completion_rate': performance_metrics['event_completion_rate']
            },
            'resources': self.resource_data.data,
            'risk_data': self.risk_model.get_risk_data(),
            'resource_flows': self.optimizer.get_resource_flows(),
            'resource_flow_summary': self.optimizer.get_resource_flow_summary()
        }
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 自定义JSON编码器，处理numpy数据类型
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        return output_file


# 如果直接运行此脚本
if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='资源优化系统')
    parser.add_argument('--algorithm', type=int, default=DEFAULT_ALGORITHM_TYPE, choices=[1, 2, 3, 4],
                        help='算法类型：1=Independent Q-Learning, 2=DQN, 3=MADDPG, 4=MAPPO')
    parser.add_argument('--data', type=str, default=None, help='初始资源数据文件路径')
    parser.add_argument('--iterations', type=int, default=AGENT_PARAMS['max_iterations'], help='最大迭代次数') # 使用config中的默认值
    parser.add_argument('--output_dir', type=str, default='data', help='输出报告目录路径') # 修改为输出目录
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 初始化系统
    system = ResourceOptimizationSystem(initial_data=None if args.data is None else args.data, algorithm_type=args.algorithm)
    
    # 如果指定了数据文件，则加载数据
    if args.data:
        system.load_data(file_path=args.data)
    
    # 执行优化
    optimization_result = system.optimize_resources(max_iterations=args.iterations)
    
    # 生成报告
    report_file_path = system.generate_report(output_dir=args.output_dir)
    
    print(f"优化完成，报告已保存至: {report_file_path}")
    # print("优化结果:")
    # print(json.dumps(optimization_result, indent=2, ensure_ascii=False, cls=NumpyEncoder)) # 使用自定义编码器打印
    
    print("\n运行完成!")