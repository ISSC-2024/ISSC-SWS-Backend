# 多智能体协同资源优化模型 - 整合版
import numpy as np
import time
import random
import os
import json
from typing import Dict, List, Any, Tuple, Optional
from collections import deque, namedtuple
from config import WORKSHOPS, RESOURCE_TYPES, RESOURCE_SUBTYPES, PRIORITY_WEIGHTS, AGENT_PARAMS, RESOURCE_INTERACTION_MATRIX, WORKSHOP_FUNCTIONS, ALGORITHM_TYPES, DEFAULT_ALGORITHM_TYPE

# 导入深度学习相关库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
TORCH_AVAILABLE = True

# 定义经验回放的数据结构
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
# 定义MAPPO经验回放的数据结构
Experience_MAPPO = namedtuple('Experience_MAPPO', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])

#######################
# MAPPO 算法实现
#######################

# MAPPO的Actor网络（策略网络）
class ActorNetwork_MAPPO(nn.Module):
    """Actor网络，用于生成策略分布"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        初始化Actor网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
        """
        super(ActorNetwork_MAPPO, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入状态
            
        Returns:
            动作概率分布
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

# MAPPO的Critic网络（价值网络）
class CriticNetwork_MAPPO(nn.Module):
    """Critic网络，用于评估状态的价值"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        """
        初始化Critic网络
        
        Args:
            state_dim: 状态空间维度
            hidden_dim: 隐藏层维度
        """
        super(CriticNetwork_MAPPO, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入状态
            
        Returns:
            状态价值估计
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 经验回放缓冲区
class PPOMemory:
    """PPO的经验回放缓冲区，用于存储和采样经验"""
    
    def __init__(self, batch_size: int):
        """
        初始化PPO经验回放缓冲区
        
        Args:
            batch_size: 批量大小
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
    
    def store(self, state, action, reward, next_state, done, log_prob, value):
        """
        存储经验
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
            log_prob: 动作的对数概率
            value: 状态的价值估计
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.ptr = (self.ptr + 1) % self.batch_size
        self.size = min(self.size + 1, self.batch_size)
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.ptr = 0
        self.size = 0
    
    def get_batch(self):
        """
        获取批量数据
        
        Returns:
            批量数据
        """
        return (
            torch.FloatTensor(self.states),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.next_states),
            torch.FloatTensor(self.dones),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.values)
        )

class ResourceAgent_MAPPO:
    """MAPPO智能体类，代表一种资源类型的分配决策者"""
    
    def __init__(self, resource_type: str, state_dim: int, action_dim: int, hidden_dim: int = 64, 
                 actor_lr: float = 3e-4, critic_lr: float = 1e-3, gamma: float = 0.99, 
                 gae_lambda: float = 0.95, clip_param: float = 0.2, value_coef: float = 0.5, 
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5,
                 initial_allocation: np.ndarray = None, load_pretrained: bool = False, 
                 model_dir: str = 'models'):
        """
        初始化MAPPO智能体
        
        Args:
            resource_type: 资源类型
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
            actor_lr: Actor网络学习率
            critic_lr: Critic网络学习率
            gamma: 折扣因子
            gae_lambda: GAE参数
            clip_param: PPO裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵损失系数
            max_grad_norm: 梯度裁剪阈值
            initial_allocation: 初始资源分配
            load_pretrained: 是否加载预训练模型
            model_dir: 模型保存目录
        """
        self.resource_type = resource_type
        self.n_workshops = state_dim  # 状态空间维度等于车间数量
        self.model_dir = model_dir
        
        # 创建模型目录
        os.makedirs(model_dir, exist_ok=True)
        
        # 初始化资源分配
        if initial_allocation is not None:
            self.allocation = initial_allocation
        else:
            # 默认平均分配
            self.allocation = np.ones(self.n_workshops) / self.n_workshops
        
        # 初始化动作映射
        self.action_map = []
        for i in range(self.n_workshops):
            for j in range(self.n_workshops):
                if i != j:
                    self.action_map.append((i, j))
        
        # MAPPO参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 初始化神经网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorNetwork_MAPPO(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork_MAPPO(state_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 初始化经验回放缓冲区
        self.memory = PPOMemory(batch_size=64)
        
        # 如果需要加载预训练模型
        if load_pretrained:
            self.load_model()
        
        # 训练历史记录
        self.training_history = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'reward': []
        }
    
    def get_state(self) -> np.ndarray:
        """
        获取当前状态的连续表示
        
        Returns:
            状态向量
        """
        # 使用连续的资源分配比例作为状态
        return self.allocation.copy()
    
    def get_actions(self) -> List[Tuple[int, int]]:
        """
        获取可能的动作列表
        
        Returns:
            动作列表，每个动作是一个元组(from_idx, to_idx)，表示从from_idx车间调整资源到to_idx车间
        """
        actions = []
        for i in range(self.n_workshops):
            for j in range(self.n_workshops):
                if i != j and self.allocation[i] > 0.05:  # 确保有足够资源可调整
                    actions.append((i, j))
        return actions
    
    def select_action(self, state: np.ndarray, available_actions: List[Tuple[int, int]] = None) -> Tuple[Tuple[int, int], float, float]:
        """
        选择动作
        
        Args:
            state: 当前状态
            available_actions: 可用动作列表
            
        Returns:
            选择的动作、动作的对数概率和状态价值
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 获取动作概率分布和状态价值
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
        
        # 如果有可用动作限制，则只考虑可用动作
        if available_actions:
            available_indices = [self.action_map.index(action) for action in available_actions]
            mask = torch.zeros(self.action_dim).to(self.device)
            mask[available_indices] = 1.0
            action_probs = action_probs * mask
            action_probs = action_probs / action_probs.sum()
        
        # 创建分类分布并采样动作
        dist = Categorical(action_probs)
        action_idx = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device)).item()
        
        # 将动作索引转换为(from_idx, to_idx)格式
        action = self.action_map[action_idx]
        
        return action, log_prob, value.item()
    
    def update_policy(self, n_epochs: int = 4):
        """
        更新策略网络和价值网络
        
        Args:
            n_epochs: 更新轮数
        
        Returns:
            更新信息字典
        """
        if len(self.memory.states) == 0:
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}
        
        states, actions, rewards, next_states, dones, old_log_probs, values = self.memory.get_batch()
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        values = values.to(self.device)
        
        # 计算优势函数
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)
        gae = 0
        
        # 计算GAE和回报
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else self.critic(next_states[t].unsqueeze(0).to(self.device)).item()
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮更新
        actor_losses = []
        critic_losses = []
        entropies = []
        
        for _ in range(n_epochs):
            # 获取新的动作概率分布和状态价值
            action_probs = self.actor(states)
            current_values = self.critic(states).squeeze()
            
            # 创建分类分布
            dist = Categorical(action_probs)
            
            # 获取动作的对数概率和熵
            action_indices = torch.tensor([self.action_map.index(tuple(map(int, a))) for a in actions]).to(self.device)
            new_log_probs = dist.log_prob(action_indices)
            entropy = dist.entropy().mean()
            
            # 计算比率和裁剪后的目标函数
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            critic_loss = F.mse_loss(current_values, returns)
            
            # 计算总损失
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # 优化模型
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.item())
        
        # 清空内存
        self.memory.clear()
        
        # 记录训练历史
        self.training_history['actor_loss'].append(np.mean(actor_losses))
        self.training_history['critic_loss'].append(np.mean(critic_losses))
        self.training_history['entropy'].append(np.mean(entropies))
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies)
        }
    
    def adjust_allocation(self, action: Tuple[int, int], adjustment_rate: float = 0.05) -> Dict[str, Any]:
        """
        调整资源分配
        
        Args:
            action: 调整动作，(from_idx, to_idx)
            adjustment_rate: 调整比率
            
        Returns:
            资源流动数据字典
        """
        from_idx, to_idx = action
        
        # 计算调整量，不超过from_idx的当前分配
        adjustment = min(self.allocation[from_idx], adjustment_rate)
        
        # 如果是人员资源类型，确保调整量为整数且最小为1人
        if self.resource_type == 'personnel' or self.resource_type.startswith('personnel_'):
            # 四舍五入为整数，并确保最小调整单位为1
            adjustment = max(1, round(adjustment))
            # 确保调整量不超过源区域的资源量
            adjustment = min(adjustment, self.allocation[from_idx])
        
        # 确保调整量为正且不超过源区域的资源量
        if adjustment <= 0 or adjustment > self.allocation[from_idx]:
            # 无法进行有效调整，返回空
            return None
            
        # 更新分配
        self.allocation[from_idx] -= adjustment
        self.allocation[to_idx] += adjustment
        
        # 确保分配总和为1
        self.allocation = self.allocation / np.sum(self.allocation)
        
        # 返回流动数据信息
        return {
            "from_workshop": from_idx,
            "to_workshop": to_idx,
            "resource_type": self.resource_type,
            "amount": float(adjustment)
        }
    
    def save_model(self) -> str:
        """
        保存模型
        
        Returns:
            模型文件路径
        """
        model_file = os.path.join(self.model_dir, f"{self.resource_type}_mappo_model")
        
        # 保存PyTorch模型
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_history': self.training_history
        }, model_file + '.pt')
        print(f"MAPPO模型已保存至: {model_file}.pt")
        return model_file + '.pt'
    
    def load_model(self) -> bool:
        """
        加载模型
        
        Returns:
            是否成功加载模型
        """
        model_file = os.path.join(self.model_dir, f"{self.resource_type}_mappo_model")
        
        # 加载PyTorch模型
        try:
            checkpoint = torch.load(model_file + '.pt', map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.training_history = checkpoint['training_history']
            print(f"已加载MAPPO模型: {model_file}.pt")
            return True
        except Exception as e:
            print(f"加载MAPPO模型失败: {e}")
            return False

#######################
# 1. Independent Q-Learning (v2) 实现
#######################

class ResourceAgent_IQL:
    """资源智能体类，代表一种资源类型的分配决策者，使用Independent Q-Learning算法"""
    
    def __init__(self, resource_type: str, initial_allocation: np.ndarray = None):
        """初始化资源智能体
        
        Args:
            resource_type: 资源类型
            initial_allocation: 初始资源分配，如果为None则平均分配
        """
        self.resource_type = resource_type
        self.n_workshops = len(WORKSHOPS)
        
        # 初始化资源分配
        if initial_allocation is not None:
            self.allocation = initial_allocation
        else:
            # 默认平均分配
            self.allocation = np.ones(self.n_workshops) / self.n_workshops
        
        # 初始化Q值表，用于强化学习
        # 状态空间：离散化的当前分配状态
        # 动作空间：资源调整方向（增加/减少）
        self.q_table = {}
        
        # 学习参数
        self.learning_rate = AGENT_PARAMS['learning_rate']
        self.discount_factor = AGENT_PARAMS['discount_factor']
        self.exploration_rate = AGENT_PARAMS['exploration_rate']
    
    def get_state(self) -> Tuple:
        """获取当前状态的离散表示
        
        Returns:
            状态元组
        """
        # 将连续的分配比例离散化为10个等级
        discrete_allocation = tuple(int(a * 10) for a in self.allocation)
        return discrete_allocation
    
    def get_actions(self) -> List[Tuple[int, int]]:
        """获取可能的动作列表
        
        Returns:
            动作列表，每个动作是一个元组(from_idx, to_idx)，表示从from_idx车间调整资源到to_idx车间
        """
        actions = []
        for i in range(self.n_workshops):
            for j in range(self.n_workshops):
                if i != j and self.allocation[i] > 0.05:  # 确保有足够资源可调整
                    actions.append((i, j))
        return actions
    
    def select_action(self, state: Tuple, available_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """选择动作
        
        Args:
            state: 当前状态
            available_actions: 可用动作列表
            
        Returns:
            选择的动作
        """
        if np.random.uniform(0, 1) < self.exploration_rate:
            # 探索：随机选择动作
            return available_actions[np.random.choice(len(available_actions))]
        else:
            # 利用：选择Q值最大的动作
            if state not in self.q_table:
                self.q_table[state] = {action: 0.0 for action in available_actions}
            
            if not self.q_table[state]:  # 如果状态对应的动作字典为空
                return available_actions[np.random.choice(len(available_actions))]
            
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, state: Tuple, action: Tuple[int, int], reward: float, next_state: Tuple) -> None:
        """更新Q值
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
        """
        if state not in self.q_table:
            self.q_table[state] = {}
        
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        # 获取下一状态的最大Q值
        if next_state in self.q_table and self.q_table[next_state]:
            max_next_q = max(self.q_table[next_state].values())
        else:
            max_next_q = 0.0
        
        # Q-learning更新公式
        self.q_table[state][action] += self.learning_rate * (
            reward + self.discount_factor * max_next_q - self.q_table[state][action]
        )
    
    def adjust_allocation(self, action: Tuple[int, int], adjustment_rate: float = 0.05) -> Dict[str, Any]:
        """调整资源分配
        
        Args:
            action: 调整动作，(from_idx, to_idx)
            adjustment_rate: 调整比率
            
        Returns:
            资源流动数据字典
        """
        from_idx, to_idx = action
        
        # 计算调整量，不超过from_idx的当前分配
        adjustment = min(self.allocation[from_idx], adjustment_rate)
        
        # 如果是人员资源类型，确保调整量为整数且最小为1人
        if self.resource_type == 'personnel' or self.resource_type.startswith('personnel_'):
            # 四舍五入为整数，并确保最小调整单位为1
            adjustment = max(1, round(adjustment))
            # 确保调整量不超过源区域的资源量
            adjustment = min(adjustment, self.allocation[from_idx])
        
        # 确保调整量为正且不超过源区域的资源量
        if adjustment <= 0 or adjustment > self.allocation[from_idx]:
            # 无法进行有效调整，返回空
            return None
            
        # 更新分配
        self.allocation[from_idx] -= adjustment
        self.allocation[to_idx] += adjustment
        
        # 确保分配总和为1
        self.allocation = self.allocation / np.sum(self.allocation)
        
        # 返回流动数据信息
        return {
            "from_workshop": WORKSHOPS[from_idx],
            "to_workshop": WORKSHOPS[to_idx],
            "resource_type": self.resource_type,
            "amount": float(adjustment)
        }


#######################
# 2. DQN (v3) 实现
#######################

# DQN网络模型定义
class DQNetwork(nn.Module):
    """深度Q网络模型，用于近似Q值函数"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """初始化DQN网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
        """
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入状态
            
        Returns:
            各动作的Q值
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ResourceAgent_DQN:
    """资源智能体类，代表一种资源类型的分配决策者，使用DQN算法"""
    
    def __init__(self, resource_type: str, initial_allocation: np.ndarray = None, load_pretrained: bool = False, model_dir: str = 'models'):
        """初始化资源智能体
        
        Args:
            resource_type: 资源类型
            initial_allocation: 初始资源分配，如果为None则平均分配
            load_pretrained: 是否加载预训练模型
            model_dir: 模型保存目录
        """
        self.resource_type = resource_type
        self.n_workshops = len(WORKSHOPS)
        self.model_dir = model_dir
        
        # 创建模型目录
        os.makedirs(model_dir, exist_ok=True)
        
        # 初始化资源分配
        if initial_allocation is not None:
            self.allocation = initial_allocation
        else:
            # 默认平均分配
            self.allocation = np.ones(self.n_workshops) / self.n_workshops
        
        # DQN参数
        self.state_dim = self.n_workshops  # 状态维度：各车间的资源分配比例
        self.action_dim = self.n_workshops * (self.n_workshops - 1)  # 动作维度：从一个车间调整到另一个车间
        self.memory_capacity = 1000  # 经验回放缓冲区容量
        self.batch_size = 32  # 批量大小
        self.target_update_freq = 10  # 目标网络更新频率
        
        # 学习参数
        self.learning_rate = AGENT_PARAMS['learning_rate']
        self.discount_factor = AGENT_PARAMS['discount_factor']
        self.exploration_rate = AGENT_PARAMS['exploration_rate']
        self.exploration_decay = 0.995  # 探索率衰减
        self.min_exploration_rate = 0.01  # 最小探索率
        
        # 初始化经验回放缓冲区
        self.memory = deque(maxlen=self.memory_capacity)
        
        # 初始化动作映射
        self.action_map = []
        for i in range(self.n_workshops):
            for j in range(self.n_workshops):
                if i != j:
                    self.action_map.append((i, j))
        
        # 初始化神经网络
        # 使用PyTorch实现
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不进行梯度更新
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # 如果需要加载预训练模型
        if load_pretrained:
            self.load_model()
        
        # 训练步数计数器
        self.steps_done = 0
        
        # 训练历史记录
        self.training_history = {
            'loss': [],
            'reward': [],
            'exploration_rate': []
        }
    
    def get_state(self) -> np.ndarray:
        """获取当前状态的连续表示
        
        Returns:
            状态向量
        """
        # 使用连续的资源分配比例作为状态
        return self.allocation.copy()
    
    def get_actions(self) -> List[Tuple[int, int]]:
        """获取可能的动作列表
        
        Returns:
            动作列表，每个动作是一个元组(from_idx, to_idx)，表示从from_idx车间调整资源到to_idx车间
        """
        actions = []
        for i in range(self.n_workshops):
            for j in range(self.n_workshops):
                if i != j and self.allocation[i] > 0.05:  # 确保有足够资源可调整
                    actions.append((i, j))
        return actions
    
    def select_action(self, state: np.ndarray, available_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """选择动作
        
        Args:
            state: 当前状态
            available_actions: 可用动作列表
            
        Returns:
            选择的动作
        """
        # 探索：随机选择动作
        if np.random.uniform(0, 1) < self.exploration_rate:
            return available_actions[np.random.choice(len(available_actions))]
        
        # 利用：选择Q值最大的动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.policy_net(state_tensor)
            
            # 过滤不可用的动作
            available_indices = [self.action_map.index(action) for action in available_actions]
            available_q_values = q_values[available_indices]
            
            # 选择Q值最大的动作
            max_q_idx = available_q_values.argmax().item()
            return available_actions[max_q_idx]
    
    def store_transition(self, state: np.ndarray, action: Tuple[int, int], reward: float, next_state: np.ndarray, done: bool) -> None:
        """存储经验到回放缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        # 将动作转换为索引
        action_idx = self.action_map.index(action)
        
        # 存储经验
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def update_model(self) -> float:
        """更新DQN模型
        
        Returns:
            损失值
        """
        # 如果经验不足，不进行更新
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # 从经验回放缓冲区中随机采样
        batch = random.sample(self.memory, self.batch_size)
        
        # 使用PyTorch实现
        states, action_idxs, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        action_idxs = torch.LongTensor(action_idxs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, action_idxs.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount_factor * max_next_q_values
        
        # 计算损失
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.clamp_(-1, 1)
        self.optimizer.step()
        
        # 更新目标网络
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def adjust_allocation(self, action: Tuple[int, int], adjustment_rate: float = 0.05) -> Dict[str, Any]:
        """调整资源分配
        
        Args:
            action: 调整动作，(from_idx, to_idx)
            adjustment_rate: 调整比率
            
        Returns:
            资源流动数据字典
        """
        from_idx, to_idx = action
        
        # 计算调整量，不超过from_idx的当前分配
        adjustment = min(self.allocation[from_idx], adjustment_rate)
        
        # 如果是人员资源类型，确保调整量为整数且最小为1人
        if self.resource_type == 'personnel' or self.resource_type.startswith('personnel_'):
            # 四舍五入为整数，并确保最小调整单位为1
            adjustment = max(1, round(adjustment))
            # 确保调整量不超过源区域的资源量
            adjustment = min(adjustment, self.allocation[from_idx])
        
        # 确保调整量为正且不超过源区域的资源量
        if adjustment <= 0 or adjustment > self.allocation[from_idx]:
            # 无法进行有效调整，返回空
            return None
            
        # 更新分配
        self.allocation[from_idx] -= adjustment
        self.allocation[to_idx] += adjustment
        
        # 确保分配总和为1
        self.allocation = self.allocation / np.sum(self.allocation)
        
        # 返回流动数据信息
        return {
            "from_workshop": WORKSHOPS[from_idx],
            "to_workshop": WORKSHOPS[to_idx],
            "resource_type": self.resource_type,
            "amount": float(adjustment)
        }
        
    def decay_exploration(self):
        """衰减探索率"""
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)
    
    def save_model(self) -> str:
        """保存模型
        
        Returns:
            模型文件路径
        """
        model_file = os.path.join(self.model_dir, f"{self.resource_type}_model")
        
        # 保存PyTorch模型
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'exploration_rate': self.exploration_rate,
            'training_history': self.training_history
        }, model_file + '.pt')
        print(f"PyTorch模型已保存至: {model_file}.pt")
        return model_file + '.pt'
    
    def load_model(self) -> bool:
        """加载模型
        
        Returns:
            是否成功加载模型
        """
        model_file = os.path.join(self.model_dir, f"{self.resource_type}_model")
        
        # 加载PyTorch模型
        try:
            checkpoint = torch.load(model_file + '.pt', map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps_done = checkpoint['steps_done']
            self.exploration_rate = checkpoint['exploration_rate']
            self.training_history = checkpoint['training_history']
            print(f"已加载PyTorch模型: {model_file}.pt")
            return True
        except Exception as e:
            print(f"加载PyTorch模型失败: {e}")
            return False


#######################
# 3. MADDPG (v4) 实现
#######################

# MADDPG的Actor网络
class ActorNetwork(nn.Module):
    """Actor网络，用于生成确定性策略（确定性动作）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """初始化Actor网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
        """
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入状态
            
        Returns:
            确定性动作（连续值）
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 使用tanh激活函数将输出限制在[-1, 1]范围内
        return torch.tanh(self.fc3(x))

# MADDPG的Critic网络
class CriticNetwork(nn.Module):
    """Critic网络，用于评估状态-动作对的价值"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """初始化Critic网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
        """
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        """前向传播
        
        Args:
            state: 输入状态
            action: 输入动作
            
        Returns:
            状态-动作对的价值估计
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 经验回放缓冲区
class ReplayBuffer:
    """经验回放缓冲区，用于存储和采样经验"""
    
    def __init__(self, capacity: int):
        """初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储经验
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """采样经验
        
        Args:
            batch_size: 批量大小
            
        Returns:
            经验批量
        """
        experiences = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.FloatTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """获取缓冲区长度
        
        Returns:
            缓冲区长度
        """
        return len(self.buffer)

class ResourceAgent_MADDPG:
    """MADDPG智能体类，代表一种资源类型的分配决策者"""
    
    def __init__(self, resource_type: str, state_dim: int, action_dim: int, hidden_dim: int = 64, 
                 actor_lr: float = 1e-4, critic_lr: float = 1e-3, gamma: float = 0.99, tau: float = 0.01,
                 initial_allocation: np.ndarray = None, load_pretrained: bool = False, model_dir: str = 'models'):
        """初始化MADDPG智能体
        
        Args:
            resource_type: 资源类型
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
            actor_lr: Actor网络学习率
            critic_lr: Critic网络学习率
            gamma: 折扣因子
            tau: 目标网络软更新系数
            initial_allocation: 初始资源分配，如果为None则平均分配
            load_pretrained: 是否加载预训练模型
            model_dir: 模型保存目录
        """
        self.resource_type = resource_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.model_dir = model_dir
        self.n_workshops = len(WORKSHOPS)
        
        # 创建模型目录
        os.makedirs(model_dir, exist_ok=True)
        
        # 初始化资源分配
        if initial_allocation is not None:
            self.allocation = initial_allocation
        else:
            # 默认平均分配
            self.allocation = np.ones(self.n_workshops) / self.n_workshops
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化Actor网络
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 初始化Critic网络
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 初始化经验回放缓冲区
        self.memory = ReplayBuffer(capacity=10000)
        
        # 探索参数
        self.exploration_rate = AGENT_PARAMS['exploration_rate']
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        
        # 训练步数计数器
        self.steps_done = 0
        
        # 训练历史记录
        self.training_history = {
            'actor_loss': [],
            'critic_loss': [],
            'reward': [],
            'exploration_rate': []
        }
        
        # 如果需要加载预训练模型
        if load_pretrained:
            self.load_model()
    
    def get_state(self) -> np.ndarray:
        """获取当前状态的连续表示
        
        Returns:
            状态向量
        """
        # 使用连续的资源分配比例作为状态
        return self.allocation.copy()
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """选择动作
        
        Args:
            state: 当前状态
            add_noise: 是否添加探索噪声
            
        Returns:
            选择的动作（资源调整向量）
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        self.actor.train()
        
        # 添加探索噪声
        if add_noise:
            noise = np.random.normal(0, self.exploration_rate, size=self.action_dim)
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action
        
    def adjust_allocation(self, action: Tuple[int, int], adjustment_rate: float = 0.05) -> Dict[str, Any]:
        """调整资源分配
        
        Args:
            action: 调整动作，(from_idx, to_idx)
            adjustment_rate: 调整比率
            
        Returns:
            资源流动数据字典
        """
        from_idx, to_idx = action
        
        # 计算调整量，不超过from_idx的当前分配
        adjustment = min(self.allocation[from_idx], adjustment_rate)
        
        # 如果是人员资源类型，确保调整量为整数且最小为1人
        if self.resource_type == 'personnel' or self.resource_type.startswith('personnel_'):
            # 四舍五入为整数，并确保最小调整单位为1
            adjustment = max(1, round(adjustment))
            # 确保调整量不超过源区域的资源量
            adjustment = min(adjustment, self.allocation[from_idx])
        
        # 确保调整量为正且不超过源区域的资源量
        if adjustment <= 0 or adjustment > self.allocation[from_idx]:
            # 无法进行有效调整，返回空
            return None
            
        # 更新分配
        self.allocation[from_idx] -= adjustment
        self.allocation[to_idx] += adjustment
        
        # 确保分配总和为1
        self.allocation = self.allocation / np.sum(self.allocation)
        
        # 返回流动数据信息
        return {
            "from_workshop": WORKSHOPS[from_idx],
            "to_workshop": WORKSHOPS[to_idx],
            "resource_type": self.resource_type,
            "amount": float(adjustment)
        }


#######################
# 4. 统一资源优化器 (v5) 实现
#######################

class ResourceOptimizer:
    """统一资源优化器类，根据选择的算法类型实例化相应的优化器"""
    
    def __init__(self, initial_resources: Dict[str, np.ndarray] = None, algorithm_type: int = DEFAULT_ALGORITHM_TYPE):
        """初始化资源优化器
        
        Args:
            initial_resources: 初始资源分配字典，键为资源类型，值为numpy数组
            algorithm_type: 算法类型，1=Independent Q-Learning, 2=DQN, 3=MADDPG, 4=MAPPO
        """
        self.algorithm_type = algorithm_type
        self.algorithm_name = ALGORITHM_TYPES.get(algorithm_type, "未知算法")
        self.agents = {}
        self.start_time = None
        self.iterations_count = 0
        self.converged = False
        # 添加资源流动数据记录字段
        self.resource_flows = []
        
        print(f"初始化资源优化器，使用算法: {self.algorithm_name}")
        
        # 创建各类资源的智能体
        for resource_type in RESOURCE_TYPES:
            initial_allocation = None
            if initial_resources and resource_type in initial_resources:
                # 归一化初始分配
                total = np.sum(initial_resources[resource_type])
                if total > 0:
                    initial_allocation = initial_resources[resource_type] / total
            
            # 根据算法类型创建不同的智能体
            if algorithm_type == 1:  # Independent Q-Learning
                self.agents[resource_type] = ResourceAgent_IQL(resource_type, initial_allocation)
            elif algorithm_type == 2:  # DQN
                self.agents[resource_type] = ResourceAgent_DQN(resource_type, initial_allocation)
            elif algorithm_type == 3:  # MADDPG
                state_dim = len(WORKSHOPS)
                action_dim = len(WORKSHOPS) * (len(WORKSHOPS) - 1)
                self.agents[resource_type] = ResourceAgent_MADDPG(
                    resource_type, state_dim, action_dim, 
                    initial_allocation=initial_allocation
                )
            elif algorithm_type == 4:  # MAPPO
                state_dim = len(WORKSHOPS)
                action_dim = len(WORKSHOPS) * (len(WORKSHOPS) - 1)
                self.agents[resource_type] = ResourceAgent_MAPPO(
                    resource_type, state_dim, action_dim, 
                    initial_allocation=initial_allocation
                )
            else:
                # 默认使用Independent Q-Learning
                print(f"警告: 未知算法类型 {algorithm_type}，使用默认算法 Independent Q-Learning")
                self.agents[resource_type] = ResourceAgent_IQL(resource_type, initial_allocation)
            
            # 创建子类型资源的智能体
            if resource_type in RESOURCE_SUBTYPES:
                for subtype in RESOURCE_SUBTYPES[resource_type]:
                    subtype_id = subtype["id"]
                    subtype_key = f"{resource_type}_{subtype_id}"
                    
                    initial_allocation = None
                    if initial_resources and subtype_key in initial_resources:
                        total = np.sum(initial_resources[subtype_key])
                        if total > 0:
                            initial_allocation = initial_resources[subtype_key] / total
                    
                    # 根据算法类型创建不同的智能体
                    if algorithm_type == 1:  # Independent Q-Learning
                        self.agents[subtype_key] = ResourceAgent_IQL(subtype_key, initial_allocation)
                    elif algorithm_type == 2:  # DQN
                        self.agents[subtype_key] = ResourceAgent_DQN(subtype_key, initial_allocation)
                    elif algorithm_type == 3:  # MADDPG
                        state_dim = len(WORKSHOPS)
                        action_dim = len(WORKSHOPS) * (len(WORKSHOPS) - 1)
                        self.agents[subtype_key] = ResourceAgent_MADDPG(
                            subtype_key, state_dim, action_dim, 
                            initial_allocation=initial_allocation
                        )
                    elif algorithm_type == 4:  # MAPPO
                        state_dim = len(WORKSHOPS)
                        action_dim = len(WORKSHOPS) * (len(WORKSHOPS) - 1)
                        self.agents[subtype_key] = ResourceAgent_MAPPO(
                            subtype_key, state_dim, action_dim, 
                            initial_allocation=initial_allocation
                        )
                    else:
                        # 默认使用Independent Q-Learning
                        self.agents[subtype_key] = ResourceAgent_IQL(subtype_key, initial_allocation)
        
        # 记录优化历史
        self.optimization_history = []
    
    def calculate_resource_interaction(self, resource_type: str, allocation: np.ndarray) -> float:
        """计算资源间的相互影响
        
        Args:
            resource_type: 资源类型
            allocation: 资源分配
            
        Returns:
            相互影响值
        """
        interaction_value = 0.0
        
        # 检查当前资源类型是否在相互影响矩阵中
        if resource_type in RESOURCE_INTERACTION_MATRIX:
            # 获取当前资源类型对其他资源类型的影响
            interactions = RESOURCE_INTERACTION_MATRIX[resource_type]
            
            # 计算当前资源分配对其他资源的影响总和
            for target_resource, impact in interactions.items():
                # 特殊处理风险影响
                if target_resource == 'risk':
                    # 负面风险影响（增加风险）会降低奖励
                    interaction_value += impact * np.sum(allocation)
                    continue
                    
                # 获取目标资源的智能体
                if target_resource in self.agents:
                    target_allocation = self.agents[target_resource].allocation
                    # 计算加权影响值（资源分配 * 目标资源分配 * 影响系数）
                    # 这里使用点积来计算两个分配之间的相互影响
                    interaction_value += impact * np.dot(allocation, target_allocation)
        
        return interaction_value
    
    def calculate_reward(self, 
                         resource_type: str, 
                         allocation: np.ndarray, 
                         risk_weights: Dict[str, float]) -> float:
        """计算奖励值
        
        Args:
            resource_type: 资源类型
            allocation: 资源分配
            risk_weights: 风险权重字典，键为车间名称，值为权重
            
        Returns:
            奖励值
        """
        reward = 0.0
        
        # 将车间名称转换为索引
        workshop_indices = {workshop: i for i, workshop in enumerate(WORKSHOPS)}
        
        # 计算资源分配与风险权重的匹配度
        for workshop, weight in risk_weights.items():
            if workshop in workshop_indices:
                idx = workshop_indices[workshop]
                # 资源分配与风险权重越匹配，奖励越高
                # 对于高风险区域，分配更多资源会获得更高奖励
                reward += allocation[idx] * weight
        
        # 考虑资源类型的优先级权重
        reward *= PRIORITY_WEIGHTS.get(resource_type, 1.0)
        
        # 考虑车间功能与资源类型的匹配度
        function_match_reward = 0.0
        for workshop, workshop_data in WORKSHOP_FUNCTIONS.items():
            if workshop in workshop_indices:
                idx = workshop_indices[workshop]
                # 获取该车间对当前资源类型的需求程度
                resource_requirements = workshop_data['resource_requirements']
                
                # 检查当前资源类型是否在该车间的需求列表中
                if resource_type in resource_requirements:
                    # 资源分配与车间功能需求越匹配，奖励越高
                    requirement_level = resource_requirements[resource_type]
                    # 计算匹配度奖励：资源分配 * 需求程度
                    function_match_reward += allocation[idx] * requirement_level
        
        # 将车间功能匹配度奖励添加到总奖励中
        reward += function_match_reward
        
        # 考虑资源间的相互影响
        interaction_value = self.calculate_resource_interaction(resource_type, allocation)
        
        # 将相互影响值添加到奖励中
        # 正向影响增加奖励，负向影响减少奖励
        reward += interaction_value
        
        return reward
    
    def optimize(self, 
                risk_weights: Dict[str, float], 
                total_resources: Dict[str, float],
                max_iterations: int = None) -> Dict[str, np.ndarray]:
        """优化资源分配
        
        Args:
            risk_weights: 风险权重字典，键为车间名称，值为权重
            total_resources: 资源总量字典，键为资源类型，值为总量
            max_iterations: 最大迭代次数，如果为None则使用配置中的值
            
        Returns:
            优化后的资源分配字典，键为资源类型，值为numpy数组
        """
        # 保存初始资源总量，用于确保资源守恒
        initial_total_resources = total_resources.copy()
        # 记录开始时间，用于计算性能指标
        self.start_time = time.time()
        
        if max_iterations is None:
            max_iterations = AGENT_PARAMS['max_iterations']
        
        convergence_threshold = AGENT_PARAMS['convergence_threshold']
        self.converged = False
        self.iterations_count = 0
        
        # 清空资源流动数据记录
        self.resource_flows = []
        
        # 记录上一次的分配结果，用于判断收敛
        previous_allocations = {resource_type: agent.allocation.copy() 
                              for resource_type, agent in self.agents.items()}
        
        while not self.converged and self.iterations_count < max_iterations:
            self.iterations_count += 1
            
            # 每个智能体进行一次决策
            for resource_type, agent in self.agents.items():
                # 根据算法类型执行不同的优化步骤
                if self.algorithm_type == 1:  # Independent Q-Learning
                    # 获取当前状态
                    state = agent.get_state()
                    
                    # 获取可用动作
                    available_actions = agent.get_actions()
                    
                    if available_actions:  # 确保有可用动作
                        # 选择动作
                        action = agent.select_action(state, available_actions)
                        
                        # 执行动作并获取流动数据
                        flow_data = agent.adjust_allocation(action)
                        if flow_data:
                            # 添加时间戳和迭代次数
                            flow_data["iteration"] = self.iterations_count
                            flow_data["timestamp"] = time.time()
                            # 如果是人员资源类型，确保流动量为整数
                            if resource_type == 'personnel' or resource_type.startswith('personnel_'):
                                flow_data["amount"] = round(flow_data["amount"])
                                
                            # 确保流动量不超过资源总量
                            if resource_type in total_resources:
                                max_flow = total_resources[resource_type] * 0.05  # 限制为总量的5%
                                if resource_type == 'personnel' or resource_type.startswith('personnel_'):
                                    max_flow = round(max_flow)
                                flow_data["amount"] = min(flow_data["amount"], max_flow)
                                
                            self.resource_flows.append(flow_data)
                        
                        # 获取新状态
                        next_state = agent.get_state()
                        
                        # 计算奖励
                        reward = self.calculate_reward(resource_type, agent.allocation, risk_weights)
                        
                        # 更新Q值
                        agent.update_q_value(state, action, reward, next_state)
                
                elif self.algorithm_type == 2:  # DQN
                    # 获取当前状态
                    state = agent.get_state()
                    
                    # 获取可用动作
                    available_actions = agent.get_actions()
                    
                    if available_actions:  # 确保有可用动作
                        # 选择动作
                        action = agent.select_action(state, available_actions)
                        
                        # 执行动作并获取流动数据
                        flow_data = agent.adjust_allocation(action)
                        if flow_data:
                            # 添加时间戳和迭代次数
                            flow_data["iteration"] = self.iterations_count
                            flow_data["timestamp"] = time.time()
                            # 如果是人员资源类型，确保流动量为整数
                            if resource_type == 'personnel' or resource_type.startswith('personnel_'):
                                flow_data["amount"] = round(flow_data["amount"])
                                
                            # 确保流动量不超过资源总量
                            if resource_type in total_resources:
                                max_flow = total_resources[resource_type] * 0.05  # 限制为总量的5%
                                if resource_type == 'personnel' or resource_type.startswith('personnel_'):
                                    max_flow = round(max_flow)
                                flow_data["amount"] = min(flow_data["amount"], max_flow)
                                
                            self.resource_flows.append(flow_data)
                        
                        # 获取新状态
                        next_state = agent.get_state()
                        
                        # 计算奖励
                        reward = self.calculate_reward(resource_type, agent.allocation, risk_weights)
                        
                        # 存储经验
                        agent.store_transition(state, action, reward, next_state, False)
                        
                        # 更新模型
                        agent.update_model()
                        
                        # 衰减探索率
                        agent.decay_exploration()
                
                elif self.algorithm_type == 3:  # MADDPG
                    # MADDPG的实现略有不同，这里是简化版
                    # 在实际应用中，MADDPG需要考虑所有智能体的联合状态和动作
                    
                    # 获取当前状态
                    state = agent.get_state()
                    
                    # 选择动作（MADDPG返回连续动作）
                    action_vector = agent.select_action(state)
                    
                    # 将连续动作转换为离散调整
                    # 这里简化处理，选择最大值对应的调整
                    action_idx = np.argmax(action_vector)
                    from_idx = action_idx // (agent.n_workshops - 1)
                    to_idx_temp = action_idx % (agent.n_workshops - 1)
                    to_idx = to_idx_temp if to_idx_temp < from_idx else to_idx_temp + 1
                    
                    # 执行动作
                    if agent.allocation[from_idx] > 0.05:  # 确保有足够资源可调整
                        # 执行动作并获取流动数据
                        flow_data = agent.adjust_allocation((from_idx, to_idx))
                        if flow_data:
                            # 添加时间戳和迭代次数
                            flow_data["iteration"] = self.iterations_count
                            flow_data["timestamp"] = time.time()
                            # 如果是人员资源类型，确保流动量为整数
                            if resource_type == 'personnel' or resource_type.startswith('personnel_'):
                                flow_data["amount"] = round(flow_data["amount"])
                            
                            # 确保流动量不超过资源总量
                            if resource_type in total_resources:
                                max_flow = total_resources[resource_type] * 0.05  # 限制为总量的5%
                                if resource_type == 'personnel' or resource_type.startswith('personnel_'):
                                    max_flow = round(max_flow)
                                flow_data["amount"] = min(flow_data["amount"], max_flow)
                                
                            self.resource_flows.append(flow_data)
                    
                    # 计算奖励
                    reward = self.calculate_reward(resource_type, agent.allocation, risk_weights)
                    
                    # 在实际MADDPG中，这里应该存储经验并更新模型
                    # 但由于简化实现，这里省略
                
                elif self.algorithm_type == 4:  # MAPPO
                    # 获取当前状态
                    state = agent.get_state()
                    
                    # 获取可用动作
                    available_actions = agent.get_actions()
                    
                    if available_actions:  # 确保有可用动作
                        # 选择动作，并获取动作的对数概率和状态价值
                        action, log_prob, value = agent.select_action(state, available_actions)
                        
                        # 执行动作并获取流动数据
                        flow_data = agent.adjust_allocation(action)
                        if flow_data:
                            # 添加时间戳和迭代次数
                            flow_data["iteration"] = self.iterations_count
                            flow_data["timestamp"] = time.time()
                            # 如果是人员资源类型，确保流动量为整数
                            if resource_type == 'personnel' or resource_type.startswith('personnel_'):
                                flow_data["amount"] = round(flow_data["amount"])
                            # 将车间索引转换为车间名称
                            flow_data["from_workshop"] = WORKSHOPS[flow_data["from_workshop"]]
                            flow_data["to_workshop"] = WORKSHOPS[flow_data["to_workshop"]]
                            
                            # 确保流动量不超过资源总量
                            if resource_type in total_resources:
                                max_flow = total_resources[resource_type] * 0.05  # 限制为总量的5%
                                if resource_type == 'personnel' or resource_type.startswith('personnel_'):
                                    max_flow = round(max_flow)
                                flow_data["amount"] = min(flow_data["amount"], max_flow)
                                
                            self.resource_flows.append(flow_data)
                        
                        # 获取新状态
                        next_state = agent.get_state()
                        
                        # 计算奖励
                        reward = self.calculate_reward(resource_type, agent.allocation, risk_weights)
                        
                        # 存储经验
                        agent.memory.store(state, action, reward, next_state, False, log_prob, value)
                        
                        # 更新策略
                        if self.iterations_count % 10 == 0:  # 每10次迭代更新一次策略
                            agent.update_policy()
            
            # 检查是否收敛
            self.converged = True
            for resource_type, agent in self.agents.items():
                diff = np.max(np.abs(agent.allocation - previous_allocations[resource_type]))
                if diff > convergence_threshold:
                    self.converged = False
                    break
            
            # 更新上一次的分配结果
            previous_allocations = {resource_type: agent.allocation.copy() 
                                  for resource_type, agent in self.agents.items()}
            
            # 记录优化历史
            self.optimization_history.append(previous_allocations.copy())
        
        # 根据循环结束的原因打印相应的信息
        if self.converged:
            print(f"已收敛。迭代次数：{self.iterations_count}，使用算法：{self.algorithm_name}")
        else:
            print(f"达到最大迭代次数：{max_iterations}，使用算法：{self.algorithm_name}")
        
        # 将比例转换为实际资源量，并确保资源总量守恒
        optimized_resources = {}
        for resource_type, agent in self.agents.items():
            if resource_type in total_resources:
                # 计算资源分配
                resource_values = agent.allocation * initial_total_resources[resource_type]
                
                # 确保所有资源值都不小于0
                resource_values = np.maximum(resource_values, 0)
                
                # 如果是人员资源类型，确保值为整数
                if resource_type == 'personnel' or resource_type.startswith('personnel_'):
                    # 四舍五入为整数
                    resource_values = np.round(resource_values)
                else:
                    # 对其他资源类型，四舍五入到整数
                    resource_values = np.round(resource_values)
                
                # 确保资源总量守恒
                current_total = np.sum(resource_values)
                original_total = initial_total_resources[resource_type]
                
                # 如果总量不一致，进行调整
                if current_total != original_total:
                    # 计算调整系数
                    if current_total > 0:
                        adjustment_factor = original_total / current_total
                        # 按比例调整所有值
                        resource_values = resource_values * adjustment_factor
                        
                        # 再次四舍五入
                        if resource_type == 'personnel' or resource_type.startswith('personnel_'):
                            resource_values = np.round(resource_values)
                        else:
                            resource_values = np.round(resource_values)
                            
                        # 处理舍入误差，确保总和精确等于原始总量
                        current_total = np.sum(resource_values)
                        if current_total != original_total:
                            # 找出最大值的索引，将差值加到或减去最大值
                            max_idx = np.argmax(resource_values)
                            resource_values[max_idx] += (original_total - current_total)
                            
                            # 确保调整后的值不为负
                            if resource_values[max_idx] < 0:
                                resource_values[max_idx] = 0
                                # 重新分配剩余差值
                                remaining_indices = [i for i in range(len(resource_values)) if i != max_idx and resource_values[i] > 0]
                                if remaining_indices:
                                    # 计算新的总和
                                    new_total = np.sum(resource_values)
                                    # 计算需要减少的量
                                    deficit = new_total - original_total
                                    # 按比例从其他正值中减去
                                    for idx in remaining_indices:
                                        proportion = resource_values[idx] / new_total
                                        resource_values[idx] -= deficit * proportion
                                        resource_values[idx] = max(0, resource_values[idx])
                                    
                                    # 最终四舍五入
                                    if resource_type == 'personnel' or resource_type.startswith('personnel_'):
                                        resource_values = np.round(resource_values)
                                    else:
                                        resource_values = np.round(resource_values)
                
                optimized_resources[resource_type] = resource_values
        
        # 将资源流动数据添加到结果中
        optimized_resources['resource_flows'] = self.resource_flows
        
        return optimized_resources
        
    def get_performance_data(self) -> Tuple[float, int, bool]:
        """获取性能数据
        
        Returns:
            包含开始时间、迭代次数和收敛状态的元组
        """
        return self.start_time, self.iterations_count, self.converged
    
    def get_optimization_history(self) -> List[Dict[str, np.ndarray]]:
        """获取优化历史
        
        Returns:
            优化历史列表
        """
        return self.optimization_history
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """获取算法信息
        
        Returns:
            算法信息字典
        """
        return {
            'type': self.algorithm_type,
            'name': self.algorithm_name
        }
        
    def get_resource_flows(self) -> List[Dict[str, Any]]:
        """获取资源流动数据
        
        Returns:
            资源流动数据列表
        """
        return self.resource_flows
        
    def get_resource_flow_summary(self) -> Dict[str, Dict[str, float]]:
        """获取资源流动总量统计
        
        Returns:
            资源流动总量统计字典，格式为{
                'resource_type': {
                    'from_workshop1_to_workshop2': total_amount,
                    ...
                },
                ...
            }
        """
        flow_summary = {}
        
        for flow in self.resource_flows:
            resource_type = flow['resource_type']
            from_workshop = flow['from_workshop']
            to_workshop = flow['to_workshop']
            amount = flow['amount']
            
            if resource_type not in flow_summary:
                flow_summary[resource_type] = {}
                
            key = f"{from_workshop}_to_{to_workshop}"
            
            if key not in flow_summary[resource_type]:
                flow_summary[resource_type][key] = 0.0
                
            flow_summary[resource_type][key] += amount
            
        return flow_summary