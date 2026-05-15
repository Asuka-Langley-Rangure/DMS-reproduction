# Overall Architecture Overview

## Project Status

当前仓库处于“**AndroidWorld 环境已接通，planner 主链路已落地，DMS 全量系统尚未完成**”的阶段。

可以按 3 类状态理解当前工程：

### Implemented

- AndroidWorld 环境可启动并用于任务初始化
- `A_zero-short_vlm_baseline.py` 可作为已有的参考基线
- `planner` 主链路已落地：
  - observation adapter
  - planner
  - OpenAI-compatible client
  - `planner_smoke.py`
  - 对应单测

### Scaffold Only

- `dms_reproduction/agents/android_actor.py`
  - 目前只有 prompt scaffold，不执行真实动作

### Planned Next

- `dms_reproduction/memory/`
- `dms_reproduction/verifier/`
- `dms_reproduction/evaluation/`

这些目录当前不是“未测试”，而是“尚未实现”。

## System Layers

当前工程可以按 5 层理解。

### 1. Environment Layer

相关路径：

- [android_world](/f:/baoyantest/dms/android_world)
- [start_androidworld_emulator.bat](/f:/baoyantest/dms/start_androidworld_emulator.bat)

作用：

- 提供 Android emulator、task registry、task initialization、env interface
- 为上层 agent 提供 `env.get_state(...)`、`task.initialize_task(...)`、`task.is_successful(...)` 等接口

输入：

- emulator / adb / grpc 端口
- AndroidWorld task id

输出：

- 可交互的 AndroidWorld env
- task goal
- UI state / screenshot / accessibility tree

当前状态：

- 已实现并可用

后续会接到：

- actor execution loop
- verifier
- evaluation runner

### 2. Observation Layer

相关路径：

- [dms_reproduction/envs/observation_utils.py](/f:/baoyantest/dms/dms_reproduction/envs/observation_utils.py)
- [dms_reproduction/envs/android_world_adapter.py](/f:/baoyantest/dms/dms_reproduction/envs/android_world_adapter.py)

作用：

- 过滤 UI element
- 生成 `ui_description`
- 生成标注图
- 构造成统一 observation schema

输入：

- AndroidWorld `state`
- `env.logical_screen_size`
- 当前 `goal`

输出：

- planner / actor / future memory 共用的 observation dict

当前状态：

- 已实现并可用于 planner smoke

后续会接到：

- `android_actor`
- memory retrieval
- verifier evidence collection

### 3. Planning Layer

相关路径：

- [dms_reproduction/agents/planner.py](/f:/baoyantest/dms/dms_reproduction/agents/planner.py)

作用：

- 做 task-level planning
- 构造 multimodal planner prompt
- 解析模型返回的 `complete_goal` / `set_tasks`

输入：

- `user_goal`
- observation
- task history
- memory context

输出：

- `PlannerResult`

当前状态：

- 已实现并可做单次真实 planner smoke

后续会接到：

- actor loop
- memory-enhanced planning
- verifier feedback loop

### 4. Model Access Layer

相关路径：

- [dms_reproduction/llm/base_client.py](/f:/baoyantest/dms/dms_reproduction/llm/base_client.py)
- [dms_reproduction/llm/openai_compatible.py](/f:/baoyantest/dms/dms_reproduction/llm/openai_compatible.py)

作用：

- 封装 vLLM / OpenAI-compatible chat completion 调用
- 把 planner messages 发给模型

输入：

- `messages`
- `temperature`
- `base_url / api_key / model / max_tokens / timeout`

输出：

- 模型原始文本响应 `str`

当前状态：

- 已实现

后续会接到：

- planner
- actor
- future verifier if it also uses the same model interface

### 5. Orchestration & Validation Layer

相关路径：

- [scripts/planner_smoke.py](/f:/baoyantest/dms/scripts/planner_smoke.py)
- [tests](/f:/baoyantest/dms/tests)

作用：

- 串起 tunnel、模拟器、AndroidWorld、adapter、planner、模型调用
- 保存 smoke artifacts
- 提供单测和 dry-run 验证

输入：

- CLI 参数
- task id
- tunnel / emulator 状态

输出：

- `planner_smoke_runs/<task_timestamp>/`
- 控制台摘要
- 单测结果

当前状态：

- 已实现，但当前只做到 planner smoke，不是正式 evaluation loop

后续会接到：

- actor execution loop
- batch evaluation
- metrics aggregation

## Current Runtime Flows

### 1. 已存在参考流程：`A_zero-short_vlm_baseline.py`

相关路径：

- [A_zero-short_vlm_baseline.py](/f:/baoyantest/dms/A_zero-short_vlm_baseline.py)

这条流程是当前理解 AndroidWorld 接口的重要参考来源。它已经实现了：

1. 加载 AndroidWorld env
2. 初始化 task
3. 抓取 screenshot 和 UI elements
4. 构造 action-level prompt
5. 直接调用 VLM
6. 解析 action JSON
7. 执行动作并循环

它的定位是：

- 历史参考实现
- 已跑通的 zero-shot baseline
- 当前新架构里很多 env / UI 处理逻辑都从这里抽取而来

但它不是后续 DMS 主线的继续堆代码位置。

### 2. 当前主流程：`scripts/planner_smoke.py`

相关路径：

- [scripts/planner_smoke.py](/f:/baoyantest/dms/scripts/planner_smoke.py)

当前新架构下，真实可运行的主流程是：

1. 建立或复用模型 tunnel
2. 检查或启动模拟器
3. 初始化 AndroidWorld task
4. observation capture
5. planner call
6. 保存结果到 `planner_smoke_runs/...`

这条流程当前只做：

- 单次 planner 调用
- 不执行 actor
- 不更新 memory
- 不做多轮 replanning

## Module Map

| Path | Responsibility | Status | Next Expected Work |
| --- | --- | --- | --- |
| `dms_reproduction/agents/planner.py` | task-level planning、message building、response parsing | Implemented | 接入 actor loop、memory context |
| `dms_reproduction/agents/android_actor.py` | Android actor 的最小 prompt scaffold | Scaffold only | 补 action parsing、真实执行、step loop |
| `dms_reproduction/envs/android_world_adapter.py` | 抓 observation 并标准化 | Implemented | 未来供 actor / memory / verifier 复用 |
| `dms_reproduction/envs/observation_utils.py` | UI 过滤、摘要、标注图、序列化 | Implemented | 根据 actor / memory 需要扩展字段 |
| `dms_reproduction/llm/openai_compatible.py` | vLLM / OpenAI-compatible HTTP 调用 | Implemented | 未来复用于 actor / verifier |
| `scripts/planner_smoke.py` | 单次真实 planner smoke orchestration | Implemented | 扩成 planner + actor 单任务闭环 |
| `tests/test_planner.py` | planner prompt 与解析单测 | Implemented | 继续覆盖更复杂的脏返回与长度约束 |
| `tests/test_planner_smoke_config.py` | smoke 配置、SSH、模拟器启动 dry-run 单测 | Implemented | 随 smoke 脚本扩展而补测试 |
| `tests/test_observation_pipeline.py` | adapter / observation schema 单测 | Implemented | 随 observation 字段变化调整 |
| `A_zero-short_vlm_baseline.py` | 已跑通的参考 zero-shot baseline | Implemented reference | 仅作参考，不作为新架构主入口 |
| `dms_reproduction/memory/` | memory 策略层预留目录 | Planned next | base interface、static memory、DMS memory |
| `dms_reproduction/verifier/` | verifier 层预留目录 | Planned next | history-first verifier、final screenshot check |

## Contracts Between Modules

### 1. Observation Contract

核心接口：

```python
AndroidWorldObservationAdapter.capture_observation(...) -> dict
```

核心字段：

- `goal`
- `current_activity`
- `app_name`
- `screen_size`
- `ui_elements`
- `ui_description`
- `valid_ui_indices`
- `screenshot_b64`
- `labeled_screenshot_b64`
- `extra_state`

说明：

- 当前 planner 使用这份 observation
- 后续 actor、memory、verifier 也应共享这份 observation contract

### 2. Planner Contract

核心接口：

```python
AndroidTaskPlanner.plan(...) -> PlannerResult
```

输出只允许两类：

- `complete_goal`
- `set_tasks`

当前也兼容解析：

- `set_tasks_with_agents`

但标准输出应以 `set_tasks` 为准。

### 3. LLM Client Contract

核心接口：

```python
generate(messages, temperature) -> str
```

说明：

- 当前 planner 依赖这一接口
- future actor 也应复用同一 contract
- 这样 planner / actor / verifier 可以独立替换底层模型实现

### 4. Smoke Artifact Contract

目录结构：

```text
planner_smoke_runs/<task_timestamp>/
```

当前落盘文件：

- `observation.json`
- `planner_messages.json`
- `planner_prompt.txt`
- `planner_raw_response.txt`
- `planner_result.json`
- `meta.json`

用途：

- 调试 prompt
- 检查模型原始返回
- 回归比较 planner 行为
- 后续为 actor / verifier / evaluation 扩展 artifact 体系

## Extension Blueprint

这一节描述如何从当前代码扩展到 DMS 全量系统。

### 1. Actor Execution Loop

目标：

- 让 planner 输出真正驱动 `android_actor`

当前状态：

- `android_actor.py` 只有 prompt scaffold

需要补：

- action parsing
- `env.execute_action(...)`
- step history
- success / failure termination
- planner 与 actor 的循环调用

### 2. Memory Layer

目标：

- 将 `null / static / dms` 设计成可替换 memory 策略

当前状态：

- `dms_reproduction/memory/` 目录存在，但没有有效实现

需要补：

- base memory interface
- static history memory
- DMS retrieval
- survival value
- pruning
- risk handling

### 3. Verifier Layer

目标：

- 引入论文中的 execution verification

当前状态：

- `dms_reproduction/verifier/` 尚未实现

需要补：

- history-first verifier prompt
- final screenshot consistency check
- planner / memory update hook

### 4. Evaluation Layer

目标：

- 稳定跑任务集并做多轮 trial 对比

当前状态：

- 尚未实现正式 evaluation loop

需要补：

- task batch runner
- metrics aggregation
- artifact organization
- SR / step count / token / memory size tracking

### 推荐开发顺序

建议后续严格按这个依赖顺序推进：

1. 先补 actor loop
2. 再抽象 memory base interface
3. 再接 verifier
4. 最后做 evaluation

原因很直接：

- 没有 actor loop，就没有真实闭环
- 没有 memory abstraction，就无法比较 `null / static / dms`
- 没有 verifier，就无法稳定做 memory 更新和风险控制
- 没有前面三层，evaluation 只会变成跑不稳定脚本

## Development Guidance

这部分是给未来继续开发时快速恢复上下文用的。

### 当前真实入口

如果要理解当前主链路，应优先从这里看起：

1. [scripts/planner_smoke.py](/f:/baoyantest/dms/scripts/planner_smoke.py)
2. [dms_reproduction/envs/android_world_adapter.py](/f:/baoyantest/dms/dms_reproduction/envs/android_world_adapter.py)
3. [dms_reproduction/agents/planner.py](/f:/baoyantest/dms/dms_reproduction/agents/planner.py)
4. [dms_reproduction/llm/openai_compatible.py](/f:/baoyantest/dms/dms_reproduction/llm/openai_compatible.py)

### 哪些地方是参考实现，不应继续堆主逻辑

- [A_zero-short_vlm_baseline.py](/f:/baoyantest/dms/A_zero-short_vlm_baseline.py)

它的定位是：

- env 接法参考
- UI 处理参考
- 历史 baseline 参考

不建议把新架构继续直接堆在这里。

### 当前 warning 的处理原则

可以先忽略：

- `Could not get a11y tree, retrying.`
  - 如果只是偶发且 observation 最终正常

需要后续正式处理：

- `Skipping app snapshot loading : Snapshot not found ...`
  - smoke 可以容忍
  - 正式 benchmark 前最好补 snapshot，否则初始状态可能漂移

### 下一步最合理的开发顺序

建议按下面顺序继续：

1. 先把 `android_actor` 从 scaffold 补成可执行动作模块
2. 再把 planner + actor 串成单任务闭环
3. 再抽象 memory base interface
4. 再接 static memory
5. 最后接 DMS-specific logic

## 与 `planner_module.md` 的分工

当前文档体系建议这样使用：

- [docs/architecture_overview.md](/f:/baoyantest/dms/docs/architecture_overview.md)
  - 先读它
  - 用来恢复全局上下文
  - 用来判断“当前做到哪里、下一步该改哪”

- [docs/planner_module.md](/f:/baoyantest/dms/docs/planner_module.md)
  - 再读它
  - 用来理解 planner 具体 API
  - 用来调试 prompt、response parsing、smoke artifacts

如果你只想恢复全局架构，先看这份文档。
如果你要继续开发 planner 或对接 actor，再看 `planner_module.md`。
