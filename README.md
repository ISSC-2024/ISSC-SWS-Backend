# 全域互联的工业智能体协同平台后端服务

## 项目简介

全域互联的工业智能体协同平台后端服务是一个基于 FastAPI 和 Tortoise-ORM 的工业监控分析平台。该系统通过四种不同的算法模型处理工业传感器数据，对化工厂各区域的安全风险进行评估和预警，并提供大模型支持的智能问答功能。

## 技术栈

- **Web 框架**：FastAPI
- **ORM**：Tortoise-ORM
- **数据库**：MySQL
- **迁移工具**：Aerich
- **异步支持**：asyncmy
- **服务器**：Uvicorn
- **AI 模型集成**：大语言模型(LLM)接口

## 项目结构

```
.
├── app                     # 应用程序主目录
│   ├── api                 # API路由定义
│   │   └── endpoints       # API端点实现
│   ├── core                # 核心配置和安全性
│   ├── data                # 数据文件目录
│   │   ├── algorithm2      # 算法2相关数据
│   │   ├── chemical_plant_dataset_1day    # 工厂日度数据集
│   │   └── chemical_plant_dataset_1hour   # 工厂小时数据集
│   ├── db                  # 数据库配置
│   ├── models              # 数据模型定义
│   │   └── algorithm2      # 算法2相关模型
│   ├── schemas             # 数据验证和序列化模型
│   │   └── algorithm2      # 算法2相关schema
│   ├── services            # 业务逻辑服务
│   │   ├── algorithm2      # 算法2相关服务
│   │   └── multiLLM        # 大语言模型服务
│   └── utils               # 工具函数
├── migrations              # 数据库迁移文件
│   └── models              # 模型迁移记录
└── tests                   # 测试代码
```

## 功能特点

1. **多算法风险评估**：支持四种不同的算法模型，用于分析和预测工业设备的安全风险
2. **知识图谱生成**：基于监测点和风险评估结果，生成工业设备关系的知识图谱
3. **智能问答**：集成大语言模型，实现基于工业数据的智能问答
4. **RESTful API**：提供完整的 API 接口，便于前端集成

## 快速开始

### 环境要求

- Python 3.8+
- MySQL 8.0+

### 安装步骤

1. 克隆代码库

   ```bash
   git clone https://github.com/yourusername/ISSC-SWS-Backend.git
   cd ISSC-SWS-Backend
   ```

2. 创建并激活虚拟环境 **（可选）**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate  # Windows
   ```

3. 安装依赖

   ```bash
   pip install -r requirements.txt
   ```

4. 配置环境变量
   创建`.env`文件，设置以下变量：

   ```
   DATABASE_URL=mysql://username:password@host:port/database
   SECRET_KEY=your_secret_key
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   ```

5. 执行数据库迁移

   ```bash
   aerich init-db
   ```

6. 启动服务

   ```bash
   uvicorn app.main:app --reload
   ```

7. 访问 API 文档：http://localhost:8000/docs

## 开发指南

### 添加新算法

1. 在 models 中创建数据库模型
2. 在 schemas 中创建相应的 Schema
3. 在 services 中实现算法逻辑
4. 在 endpoints 中实现 API
5. 在`app/main.py`中注册路由

### 数据库迁移

创建新的迁移：

```bash
aerich migrate --name migration_name
```

应用迁移：

```bash
aerich upgrade
```
