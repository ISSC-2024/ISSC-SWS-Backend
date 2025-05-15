# 多智能体协同资源优化系统

## 系统概述

本系统是一个基于多智能体强化学习的资源优化分配系统，用于模拟和优化工业环境中的资源分配问题。系统通过多种强化学习算法，根据风险评估数据，自动调整各车间区域的资源分配，以实现资源利用最大化和风险最小化。

## 文件结构

- `main.py`: 主系统模块，整合各个子模块并提供命令行接口
- `config.py`: 配置文件，包含系统参数、车间信息、资源类型等配置
- `data_model.py`: 数据模型模块，处理资源数据的加载、存储和转换
- `risk_model.py`: 风险模型模块，处理风险评估数据和计算风险权重
- `optimization_model.py`: 优化模型模块，实现多种强化学习算法进行资源优化

## 使用方法

### 命令行参数

系统可以通过命令行参数或者config文件里的参数进行配置和运行：

```bash
python main.py [--algorithm ALGORITHM] [--data DATA_FILE] [--iterations ITERATIONS] [--output_dir OUTPUT_DIR]
```

参数说明：

- `--algorithm`: 算法类型，整数值，可选值为1-4
  - 1: Independent Q-Learning (IQL)[默认]
  - 2: Deep Q-Network (DQN)
  - 3: Multi-Agent Deep Deterministic Policy Gradient (MADDPG) 
  - 4: Multi-Agent Proximal Policy Optimization (MAPPO)

- `--data`: 初始资源数据文件路径，JSON格式
  - 如果不提供，系统将使用默认路径`data/initial_resources.json`
  - 如果文件不存在，系统将使用默认的平均分配

- `--iterations`: 最大迭代次数
  - 默认值从`config.py`中的`AGENT_PARAMS['max_iterations']`获取（默认为1000）

- `--output_dir`: 输出报告目录路径
  - 默认为`data`目录

### 示例

```bash
# 使用MAPPO算法，最大迭代次数为500
python main.py --algorithm 4 --iterations 500

# 使用MADDPG算法，从指定文件加载初始数据
python main.py --algorithm 3 --data data/custom_resources.json

# 使用默认算法，输出到指定目录
python main.py --output_dir results/test1
```

## 配置文件参数

系统的主要配置参数位于`config.py`文件中：

### 智能体参数 (AGENT_PARAMS)

```python
AGENT_PARAMS = {
    "learning_rate": 0.01,           # 学习率
    "discount_factor": 0.95,        # 折扣因子
    "exploration_rate": 0.1,        # 探索率
    "max_iterations": 1000,         # 最大迭代次数
    "convergence_threshold": 0.005  # 收敛阈值
}
```

- `learning_rate`: 智能体学习率，影响学习速度和稳定性
- `discount_factor`: 折扣因子，控制未来奖励的重要性
- `exploration_rate`: 探索率，控制智能体探索新动作的概率
- `max_iterations`: 最大迭代次数，防止无限循环
- `convergence_threshold`: 收敛阈值，当资源分配变化小于此值时认为已收敛

### 其他重要配置

- `WORKSHOPS`: 车间列表，定义系统中的车间区域
- `RESOURCE_TYPES`: 资源类型列表
- `RESOURCE_SUBTYPES`: 资源子类型定义
- `PRIORITY_WEIGHTS`: 资源优先级权重
- `WORKSHOP_FUNCTIONS`: 车间功能与资源需求定义
- `RESOURCE_INTERACTION_MATRIX`: 资源间相互影响矩阵
- `DEFAULT_ALGORITHM_TYPE`: 默认算法类型

## 编程接口

### 主类：ResourceOptimizationSystem

```python
system = ResourceOptimizationSystem(initial_data=None, algorithm_type=DEFAULT_ALGORITHM_TYPE)
```

参数：
- `initial_data`: 初始资源数据，可以是字典或文件路径
- `algorithm_type`: 算法类型，整数值1-4

### 主要方法

```python
# 加载资源数据
system.load_data(file_path=None, data=None)

# 保存资源数据
system.save_data(file_path)

# 更新风险数据
system.update_risk_data(new_risk_data=None)

# 优化资源分配
optimization_result = system.optimize_resources(max_iterations=1000)

# 生成报告
report_file_path = system.generate_report(output_dir='data')
```

## 输出格式

### 优化结果

优化结果是一个包含以下字段的字典：

```python
{
    'optimized_resources': {...},  # 优化后的资源分配
    'risk_data': {...},            # 风险数据
    'resource_data': {...},        # 资源数据
    'performance_metrics': {...},  # 性能指标
    'algorithm_type': int,         # 算法类型
    'algorithm_name': str,         # 算法名称
    'resource_flows': [...],       # 资源流动数据
    'resource_flow_summary': {...} # 资源流动总量统计
}
```

### 输出文件

系统会在指定的输出目录（默认为`data`）生成JSON格式的报告文件，文件名格式为：

```
{algorithm_name}__{convergence_threshold}_allocate_result.json
```

例如：`MADDPG__0.005_allocate_result.json`

报告文件包含以下内容：

```json
{
  "timestamp": "2023-01-01 12:00:00",
  "algorithm": {
    "type": 3,
    "name": "MADDPG"
  },
  "performance": {
    "execution_time": 1.234,
    "iterations": 100,
    "converged": true,
    "response_time": 1234.5,
    "response_timeliness": 0.95,
    "response_quality": 0.85,
    "resource_utilization": 0.9,
    "event_completion_rate": 0.88
  },
  "resources": {...},
  "risk_data": {...},
  "resource_flows": [...],
  "resource_flow_summary": {...}
}
```

## 算法说明

系统实现了四种强化学习算法：

1. **Independent Q-Learning (IQL)**: 基础的Q学习算法，每个智能体独立学习
2. **Deep Q-Network (DQN)**: 使用深度神经网络的Q学习算法
3. **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**: 多智能体确定性策略梯度算法
4. **Multi-Agent Proximal Policy Optimization (MAPPO)**: 多智能体近端策略优化算法

每种算法都有其特点和适用场景，可以根据具体需求选择合适的算法。

## 注意事项

1. 系统需要Python 3.6+环境
2. 需要安装PyTorch、NumPy等依赖库
3. 首次运行时会自动创建`data`和`models`目录
4. 风险数据默认从`data/predicted_results.csv`读取，如果文件不存在，将使用随机数据
5. 模型文件保存在`models`目录下