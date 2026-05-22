# DMS复现项目说明

## 1. 项目整体目标

本项目是在 **AndroidWorld** 环境上，对 **Darwinian Memory System (DMS)** 思路进行工程化复现与实验验证。

整体系统由以下几部分组成：

- AndroidWorld 环境接入
- Planner / Actor / Verifier 智能体主流程
- DMS 记忆模块
- 模型调用管道
- 自动化运行与结果汇总脚本

---

## 2. 项目主体结构

### `android_world/`

这一部分是 **AndroidWorld 环境代码**，主要负责：

- Android 模拟器 / 任务环境接入
- 任务定义
- 环境执行与评测接口

可以理解为：这是本项目运行所依赖的“实验环境层”。

### `dms_reproduction/`

这是本项目的**核心复现代码**，主要分为以下模块：

#### `dms_reproduction/agents/`

智能体主流程代码：

- `planner.py`
  - 负责根据当前界面状态把用户目标拆成 subtasks
- `android_actor.py`
  - 负责执行单个 subtask，对界面进行点击、输入、滚动等操作
- `verifier.py`
  - 负责判断一个 subtask 是否真正完成
- `task_runner.py`
  - 负责组织 planner → actor → verifier / evaluator 的完整循环

#### `dms_reproduction/memory/`

记忆系统相关代码：

- `static.py`
  - 静态记忆检索
- `dms.py`
  - DMS 记忆读写与调度逻辑
- `retrieval.py`
  - 记忆检索打分逻辑
- `pruning.py`
  - 记忆剪枝逻辑
- `survival_value.py`
  - 生存价值计算
- `store.py`
  - 记忆存储层

这一部分是本项目和普通无记忆 agent 的主要区别。

#### `dms_reproduction/llm/`

模型调用相关代码：

- `openai_compatible.py`
  - 对接 OpenAI-compatible 接口
- `base_client.py`
  - usage、配置与通用封装

#### `dms_reproduction/envs/`

环境适配层：

- 将 AndroidWorld 的 observation 转成当前 agent 可直接使用的格式
- 负责截图、UI 元素表、界面摘要等信息整理

---

## 3. `scripts/` 目录

这一部分是**运行脚本层**，主要用于实验执行和结果统计。

常用脚本包括：

- `task_loop_smoke.py`
  - 单任务运行入口
- `run_task_loop_batch.py`
  - 单任务多次重复运行
- `aggregate_batch_results.py`
  - 汇总 batch 结果并导出 CSV / JSON

可以理解为：这是本项目的“实验驱动层”。

---

## 4. `memory_bank/` 目录

这一部分用于保存记忆系统的实际数据，例如：

- static memory 文件
- dms memory 根目录

如果运行 `memory_backend=none`，则不会真正使用这一部分。

---

## 5. 三种实验设置

本项目目前支持三种实验线：

- `memory_backend=none`
  - 无记忆基线
- `memory_backend=static`
  - 静态记忆检索
- `memory_backend=dms`
  - Darwinian Memory System 动态记忆

