# 配置文件
import numpy as np

# 车间列表
WORKSHOPS = [
    "原料储存区",  # RMS
    "反应器区",    # REA
    "分离提纯区",  # SEP
    "成品储存区",  # PRO
    "公用工程区"   # UTL
]

# 资源类型
RESOURCE_TYPES = [
    "personnel",    # 人力资源
    "materials",    # 物料
    "electricity"   # 电力
]

# 资源子类型
RESOURCE_SUBTYPES = {
    "personnel": [
        {"id": "technician", "name": "技术员"},
        {"id": "maintenance", "name": "维修员"},
        {"id": "manager", "name": "管理员"},
        {"id": "operator", "name": "操作员"},
        {"id": "safety", "name": "安全员"}
    ],
    "materials": [
        {"id": "raw", "name": "原料"},
        {"id": "catalyst", "name": "催化剂"},
        {"id": "storage", "name": "存储容量"}
    ]
}

# 资源优先级权重
PRIORITY_WEIGHTS = {
    "electricity": 0.8,
    "personnel": 1.0,
    "materials": 0.9,
    "personnel_technician": 1.0,
    "personnel_maintenance": 1.1,
    "personnel_manager": 1.2,
    "personnel_operator": 0.9,
    "personnel_safety": 1.1,
    "materials_raw": 0.9,
    "materials_catalyst": 1.1,
    "materials_storage": 0.8
}

# 车间功能与资源需求
WORKSHOP_FUNCTIONS = {
    "原料储存区": {  # RMS
        "function": "原料储存区",
        "code": "RMS",
        "resource_requirements": {
            "electricity": 0.7,
            "personnel": 1.0,
            "materials": 1.0,
            "personnel_technician": 1.0,
            "personnel_maintenance": 0.8,
            "personnel_manager": 0.6,
            "personnel_operator": 1.2,
            "personnel_safety": 0.7,
            "materials_raw": 1.2,
            "materials_catalyst": 0.5,
            "materials_storage": 1.0
        }
    },
    "反应器区": {  # REA
        "function": "反应器区",
        "code": "REA",
        "resource_requirements": {
            "electricity": 1.0,
            "personnel": 0.8,
            "materials": 0.9,
            "personnel_technician": 0.9,
            "personnel_maintenance": 0.7,
            "personnel_manager": 0.5,
            "personnel_operator": 1.0,
            "personnel_safety": 0.6,
            "materials_raw": 0.8,
            "materials_catalyst": 1.2,
            "materials_storage": 0.7
        }
    },
    "分离提纯区": {  # SEP
        "function": "分离提纯区",
        "code": "SEP",
        "resource_requirements": {
            "electricity": 1.2,
            "personnel": 0.6,
            "materials": 0.7,
            "personnel_technician": 0.8,
            "personnel_maintenance": 0.6,
            "personnel_manager": 0.4,
            "personnel_operator": 0.7,
            "personnel_safety": 0.5,
            "materials_raw": 0.6,
            "materials_catalyst": 1.0,
            "materials_storage": 1.1
        }
    },
    "成品储存区": {  # PRO
        "function": "成品储存区",
        "code": "PRO",
        "resource_requirements": {
            "electricity": 0.9,
            "personnel": 0.7,
            "materials": 0.8,
            "personnel_technician": 0.7,
            "personnel_maintenance": 0.5,
            "personnel_manager": 0.6,
            "personnel_operator": 0.9,
            "personnel_safety": 0.5,
            "materials_raw": 0.5,
            "materials_catalyst": 0.4,
            "materials_storage": 1.2
        }
    },
    "公用工程区": {  # UTL
        "function": "公用工程区",
        "code": "UTL",
        "resource_requirements": {
            "electricity": 0.8,
            "personnel": 0.9,
            "materials": 0.5,
            "personnel_technician": 1.2,
            "personnel_maintenance": 1.1,
            "personnel_manager": 0.8,
            "personnel_operator": 0.6,
            "personnel_safety": 0.7,
            "materials_raw": 0.4,
            "materials_catalyst": 0.6,
            "materials_storage": 0.5
        }
    }
}

# 资源间相互影响矩阵
RESOURCE_INTERACTION_MATRIX = {
    "electricity": {
        "personnel": 0.2,  # 电力对人员有正面影响
        "materials": 0.3,  # 电力对物料有正面影响
        "risk": 0.1       # 电力对风险有轻微正面影响（增加风险）
    },
    "personnel": {
        "electricity": 0.1,  # 人员对电力有轻微正面影响
        "materials": 0.4,   # 人员对物料有较强正面影响
        "risk": -0.2       # 人员对风险有负面影响（降低风险）
    },
    "materials": {
        "electricity": 0.0,  # 物料对电力无影响
        "personnel": 0.1,   # 物料对人员有轻微正面影响
        "risk": 0.05       # 物料对风险有轻微正面影响（增加风险）
    },
    "personnel_technician": {
        "electricity": 0.15,  # 技术员对电力有正面影响
        "materials": 0.25,    # 技术员对物料有正面影响
        "risk": -0.25        # 技术员对风险有负面影响（降低风险）
    },
    "personnel_maintenance": {
        "electricity": 0.2,   # 维修员对电力有正面影响
        "materials": 0.2,     # 维修员对物料有正面影响
        "risk": -0.3         # 维修员对风险有较强负面影响（降低风险）
    },
    "personnel_manager": {
        "electricity": 0.1,   # 管理员对电力有轻微正面影响
        "materials": 0.3,     # 管理员对物料有正面影响
        "risk": -0.2         # 管理员对风险有负面影响（降低风险）
    },
    "personnel_operator": {
        "electricity": 0.05,  # 操作员对电力有轻微正面影响
        "materials": 0.35,    # 操作员对物料有较强正面影响
        "risk": -0.15        # 操作员对风险有负面影响（降低风险）
    },
    "personnel_safety": {
        "electricity": 0.1,   # 安全员对电力有轻微正面影响
        "materials": 0.15,    # 安全员对物料有轻微正面影响
        "risk": -0.35        # 安全员对风险有强负面影响（降低风险）
    },
    "materials_raw": {
        "electricity": 0.05,  # 原料对电力有轻微正面影响
        "personnel": 0.05,    # 原料对人员有轻微正面影响
        "risk": 0.1          # 原料对风险有轻微正面影响（增加风险）
    },
    "materials_catalyst": {
        "electricity": 0.15,  # 催化剂对电力有正面影响
        "personnel": 0.1,     # 催化剂对人员有轻微正面影响
        "risk": 0.2          # 催化剂对风险有正面影响（增加风险）
    },
    "materials_storage": {
        "electricity": 0.1,   # 存储容量对电力有轻微正面影响
        "personnel": 0.0,     # 存储容量对人员无影响
        "risk": 0.05         # 存储容量对风险有轻微正面影响（增加风险）
    }
}

# 默认输入文件路径
INPUT_FILE_PATH = 'data/initial_resources.json'

# 风险数据文件路径
RISK_DATA_FILE_PATH = 'data/predicted_results.csv'

# 智能体参数
AGENT_PARAMS = {
    "learning_rate": 0.01,           # 学习率
    "discount_factor": 0.95,        # 折扣因子
    "exploration_rate": 0.1,        # 探索率
    "max_iterations": 1000,         # 最大迭代次数
    "convergence_threshold": 0.005  # 收敛阈值
}


# 算法类型配置
ALGORITHM_TYPES = {
    1: "Independent Q-Learning",  # v2版本算法
    2: "DQN",                    # v3版本算法
    3: "MADDPG",                 # v4版本算法
    4: "MAPPO"                   # v5版本算法
}

# 默认算法类型
DEFAULT_ALGORITHM_TYPE = 1