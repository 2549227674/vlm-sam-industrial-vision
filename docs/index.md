# 项目文档索引

**项目名称**：SAM-3 双范式工业视觉分析系统
**生成日期**：2026-01-03
**文档版本**：v1.0
**扫描模式**：快速扫描（基础文档）

---

## 📋 项目概览

### 基本信息
- **项目类型**：数据科学/机器学习应用（Streamlit Web App）
- **架构模式**：单体应用（Monolith）
- **主要语言**：Python 3.8+
- **核心技术**：PyTorch, Streamlit, SAM-3, VLM, PaDiM
- **部署模式**：本地运行

### 项目定位
基于 VLM 语义引导与 SAM-3 开放词汇模型的工业视觉分析系统，支持三种分析范式：
- **范式 A**：VLM 引导的开放词汇实例分割
- **范式 B**：SAM-3 Purify + PaDiM 离线异常检测
- **范式 C**：VLM 缺陷框 → SAM 精分割

### 未来升级方向
**边缘-云协同工业视觉检测系统**：
- **边缘节点**：i.MX6ULL（Linux）- 视频采集 + 云台控制
- **云节点**：PC（RTX 4060）- VLM/SAM 推理计算
- **通信**：TCP/IP Socket（以太网）

---

## 📚 文档导航

### 核心文档

#### 1. [技术栈分析](./technology-stack.md)
**内容**：
- 完整的技术栈清单（框架、库、工具）
- 三范式架构说明
- 模型列表与用途
- 代码结构概览
- 扩展方向（边缘-云协同）

**适用场景**：
- 了解项目使用的技术
- 评估技术选型
- 规划技术升级

---

#### 2. [源代码树分析](./source-tree-analysis.md)
**内容**：
- 完整的目录结构（带注释）
- 模块职责说明
- 数据流架构
- 模块依赖关系
- 入口点与启动流程

**适用场景**：
- 快速了解代码组织
- 定位功能模块
- 理解模块间关系

---

#### 3. [系统架构文档](./architecture.md)
**内容**：
- 整体架构设计（分层架构）
- 三范式详细架构
- 数据流设计
- 模型管理策略
- 部署架构（当前 + 未来）
- 性能特性
- 可扩展性设计

**适用场景**：
- 理解系统设计
- 架构评审
- 规划系统升级

---

#### 4. [开发指南](./development-guide.md)
**内容**：
- 环境准备与安装
- 依赖安装（完整/最小化）
- 环境变量配置
- 运行应用
- 开发工作流
- 调试技巧
- 常见问题解决
- 性能优化
- 部署指南

**适用场景**：
- 新开发者入门
- 环境搭建
- 日常开发参考
- 问题排查

---

### 辅助文档

#### 5. [项目扫描报告](./project-scan-report.json)
**内容**：
- 扫描元数据
- 项目分类信息
- 用户上下文
- 扫描进度记录

**适用场景**：
- 工作流恢复
- 扫描状态追踪

---

## 🚀 快速开始

### 1. 环境准备
```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 安装依赖
cd 计算机视觉项目
pip install -r requirements.txt
```

### 2. 配置（可选）
```bash
# 启用 VLM 功能
set DASHSCOPE_API_KEY=sk-your-key-here  # Windows
# export DASHSCOPE_API_KEY=sk-your-key-here  # Linux/macOS
```

### 3. 运行应用
```bash
streamlit run app_final.py
```

### 4. 访问应用
浏览器打开：`http://localhost:8501`

---

## 🎯 使用场景指南

### 场景 1：新开发者入门
**推荐阅读顺序**：
1. [技术栈分析](./technology-stack.md) - 了解技术选型
2. [源代码树分析](./source-tree-analysis.md) - 熟悉代码结构
3. [开发指南](./development-guide.md) - 搭建环境并运行

### 场景 2：功能开发
**推荐阅读顺序**：
1. [系统架构文档](./architecture.md) - 理解整体设计
2. [源代码树分析](./source-tree-analysis.md) - 定位相关模块
3. [开发指南](./development-guide.md) - 参考开发工作流

### 场景 3：系统升级（边缘-云协同）
**推荐阅读顺序**：
1. [系统架构文档](./architecture.md) - 查看未来部署架构
2. [技术栈分析](./technology-stack.md) - 查看扩展方向
3. [源代码树分析](./source-tree-analysis.md) - 查看扩展点

### 场景 4：问题排查
**推荐阅读顺序**：
1. [开发指南](./development-guide.md) - 查看常见问题
2. [系统架构文档](./architecture.md) - 理解系统行为
3. [源代码树分析](./source-tree-analysis.md) - 定位问题模块

---

## 📊 项目统计

### 代码规模
- **Python 文件**：25 个
- **核心模块**：14 个文件（core/）
- **UI 模块**：10 个文件（ui/）
- **主入口**：1 个文件（app_final.py）

### 关键模块大小
- `ui/paradigm_a.py`：61KB（范式A界面）
- `ui/paradigm_c.py`：55KB（范式C界面）
- `ui/styles.py`：24KB（样式系统）
- `ui/paradigm_b.py`：23KB（范式B界面）
- `ui/components.py`：22KB（通用组件）
- `core/vlm_bbox.py`：22KB（VLM边界框）

### 技术栈
- **UI 框架**：Streamlit ≥1.36.0
- **深度学习**：PyTorch ≥2.0.0
- **计算机视觉**：OpenCV ≥4.8.0
- **模型仓库**：ModelScope ≥1.9.0
- **VLM API**：DashScope ≥1.14.0

---

## 🔗 外部资源

### 官方文档
- [Streamlit 文档](https://docs.streamlit.io)
- [PyTorch 文档](https://pytorch.org/docs)
- [ModelScope 文档](https://modelscope.cn/docs)
- [DashScope API](https://dashscope.aliyuncs.com)

### 模型资源
- [SAM-3 模型](https://modelscope.cn/models)
- [ResNet 模型](https://pytorch.org/vision/stable/models.html)

---

## 📝 文档维护

### 文档生成信息
- **生成工具**：BMAD Document Project Workflow
- **生成时间**：2026-01-03
- **扫描模式**：快速扫描（基础文档）
- **项目根目录**：`C:\Users\Han\PycharmProjects\PythonProject8\计算机视觉项目`

### 文档更新
当项目发生重大变更时，建议重新运行文档生成工作流：
```bash
# 在 BMAD 系统中运行
/bmad:bmm:workflows:document-project
```

---

## 🎓 下一步行动

### Phase 1：分析阶段（可选）
- **头脑风暴**：运行 `/bmad:core:workflows:brainstorming` 探索边缘-云协同方案
- **研究**：运行 `/bmad:bmm:workflows:research` 进行技术调研

### Phase 2：规划阶段
- **PRD**：运行 `/bmad:bmm:workflows:create-prd` 创建产品需求文档
  - 输入：当前项目文档 + 边缘-云协同需求
  - 输出：详细的功能需求和非功能需求

### Phase 3：设计阶段
- **架构设计**：运行 `/bmad:bmm:workflows:create-architecture` 设计边缘-云架构
  - 输入：PRD + 当前架构文档
  - 输出：分布式系统架构设计

### Phase 4：实现阶段
- **Epics & Stories**：运行 `/bmad:bmm:workflows:create-epics-and-stories` 分解任务
- **开发**：按 Story 逐步实现功能

---

## 💡 重要提示

### 当前系统状态
✅ **已完成**：Phase 0 - 项目文档化
⏳ **下一步**：Phase 1 - 分析阶段（头脑风暴/研究）

### 文档使用建议
1. **开发前**：先阅读架构文档和源代码树分析
2. **开发中**：参考开发指南和技术栈文档
3. **问题排查**：查看开发指南的常见问题部分
4. **系统升级**：参考架构文档的未来部署模式

### 文档反馈
如发现文档错误或需要补充，请：
1. 记录问题描述
2. 在下次文档生成时提供反馈
3. 或手动更新相关文档

---

**文档索引生成时间**：2026-01-03
**文档总数**：5 个核心文档
**项目状态**：已完成基础文档化，准备进入分析阶段
