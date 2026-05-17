# Android GUI PA Agent 项目阶段总结

## 1. 项目目标与当前定位

本项目基于 `AndroidWorld` 复现一个 Android GUI Planner-Actor（PA）agent 系统，并为后续接入静态记忆系统与 DMS 记忆系统预留接口。

当前代码主线已经从最初的单轮 planner smoke，推进到一个可运行的闭环：

`planner -> android_actor -> task_runner -> AndroidWorld evaluator -> smoke artifacts`

当前项目的真实状态可以概括为：

- `AndroidWorld` 环境适配与 observation 标准化：已实现
- `planner`：已实现，支持通用 prompt 与 legacy 对照 prompt
- `android_actor`：已实现，支持多步 GUI 动作执行
- `task_runner`：已实现，承担主编排、replan、子任务成功判定和部分任务特化保护
- `memory`：仅完成接口预留，默认是 no-op
- `verifier`：当前源码树中没有稳定的可追踪实现，尚未纳入正式主链路
- 真实 smoke 跑通的主要任务仍集中在 `ContactsAddContact`

这意味着项目已经进入“闭环可运行，但泛化与稳定性不足”的阶段，而不是“仅有 scaffold”阶段。

## 2. 当前目录结构与职责

下面只列当前对 PA agent 主链路最关键的目录和文件。

### 2.1 运行时核心代码

- `dms_reproduction/agents/planner.py`
  - 任务级 planner
  - 负责根据总任务、当前 observation、task history、memory context 产出 `1-5` 个短周期子任务
  - 输出格式固定为 `Precondition: ... Goal: ...`

- `dms_reproduction/agents/android_actor.py`
  - 子任务执行器
  - 负责针对一个当前 subtask，在当前 screen observation 上逐步生成 GUI action
  - 支持 `click / input_text / scroll / open_app / navigate_back / wait / status` 等动作

- `dms_reproduction/agents/task_runner.py`
  - 主编排层
  - 负责：
    - 调 planner
    - 调 actor
    - observation refresh
    - 子任务成功 override
    - contact 场景下的 grouped form postprocess
    - task history 写回
    - memory hook 写回
    - 最终调用 AndroidWorld evaluator

### 2.2 环境与 observation

- `dms_reproduction/envs/android_world_adapter.py`
  - 从 AndroidWorld `env` 中采样统一 observation
  - 负责截图、带 index 的截图、UI elements、前台 activity/package、稳定性判断

- `dms_reproduction/envs/observation_utils.py`
  - 负责 UI element 标准化、UI description 构造、可视元素筛选、截图标号

### 2.3 模型访问层

- `dms_reproduction/llm/base_client.py`
  - OpenAI-compatible 配置结构

- `dms_reproduction/llm/openai_compatible.py`
  - 通过 HTTP 请求 OpenAI-compatible chat completion 接口
  - 当前 planner 和 actor 共用这一路模型调用方式

### 2.4 记忆接口

- `dms_reproduction/memory/base.py`
  - 定义 `MemoryProvider` 协议
  - 定义 `MemoryEvent`
  - 提供默认 `NoOpMemoryProvider`

当前这里的意义是“为后续静态记忆/DMS 接线”，不是“已有记忆能力”。

### 2.5 运行脚本与评测脚本

- `scripts/task_loop_smoke.py`
  - 单次真实闭环 smoke 入口
  - 负责启动 planner / actor / runner，并产出完整 artifacts

- `scripts/prompt_eval_smoke.py`
  - 独立批量 smoke 评测脚本
  - 当前用于“同一修改连续跑 3 次并汇总结果”
  - 这部分评测规则在脚本里，而不是固化在主系统逻辑里

### 2.6 文档与测试

- `README.md`
  - 仓库入口说明

- `docs/current_status.md`
  - 当前阶段状态记录

- `docs/architecture_overview.md`
  - 原本用于架构概览，但当前文件存在明显编码问题，内容不适合作为对外同步主文档

- `tests/`
  - 已有较完整的 mock/unit tests，覆盖 planner、actor、task_runner、observation pipeline、smoke artifact、prompt eval 脚本

## 3. 各模块实现原理

## 3.1 Observation 层

`AndroidWorldObservationAdapter.capture_observation(...)` 是整个系统的统一输入入口。

它从 AndroidWorld 读出：

- 当前像素截图
- UI accessibility elements
- 当前前台 activity
- 屏幕大小

然后进一步加工为统一 observation dict，核心字段包括：

- `current_activity`
- `foreground_package`
- `app_name`
- `ui_elements`
- `ui_description`
- `valid_ui_indices`
- `visible_ui_count`
- `clickable_ui_count`
- `editable_ui_count`
- `non_system_ui_count`
- `observation_warning`
- `observation_consistency`
- `screenshot_b64`
- `labeled_screenshot_b64`

当前 observation 层的一个关键设计点是：它会尝试重采样，以尽量拿到 `stable` observation；如果多次仍不稳定，则把“不稳定状态”显式传给上层，而不是假装屏幕可靠。

## 3.2 Planner 层

`AndroidTaskPlanner` 的职责不是直接产出动作，而是产出短周期功能性子任务。

当前 planner 的主要机制：

- 输入：
  - `user_goal`
  - `observation`
  - `task_history`
  - `memory_context`
- 输出：
  - `PlannerResult`
  - 若完成则 `complete_goal`
  - 否则返回 `set_tasks`

当前 planner 已实现两套 prompt profile：

- `generic_dms`
  - 默认 profile
  - 尽量保持通用，不把具体任务步骤硬编码进 prompt

- `legacy_contact_tuned`
  - 历史对照 profile
  - 保留了一些更偏向 contacts 任务的旧式调优逻辑

planner 解析层还包含：

- JSON 提取与修复
- `Precondition: ... Goal: ...` 格式约束
- `PlannerSubtask` 结构化表示

因此 planner 层的设计原则是：

- 只做“下一阶段应该达到什么状态”的规划
- 不直接写成点击级别脚本
- 每轮都看当前 observation 和历史重新规划

## 3.3 Actor 层

`AndroidActor` 负责把一个 subtask 执行成一串 GUI 动作。

运行逻辑是：

1. 接收一个 `ActorRequest`
2. 基于当前 observation、history、memory context 构造 prompt
3. 调用 LLM 生成一条 `Reason + Action`
4. 解析为结构化 action
5. 执行动作
6. 重新采 observation
7. 继续下一步，直到：
   - `completed`
   - `infeasible`
   - `step_limit`
   - `parse_error`
   - `execution_error`

当前 actor 的特点：

- 已支持完整动作 schema
- 已有动作 normalization
- 已有 index correction
- 已有 observation degraded 场景下的保守处理

其中一个重要的工程点是：actor 输出只是“候选动作”，`task_runner` 仍会做额外校验与纠偏，因此 actor 不是最终真值来源。

## 3.4 Task Runner 层

`AndroidTaskRunner` 是目前系统里最关键的模块，也是当前最“重”的一层。

它承担了四类职责：

### A. 闭环编排

- 初始化 task
- 首次采 observation
- 调 planner
- 逐个调 actor 执行 subtask
- 根据结果决定：
  - 继续当前 round
  - 提前 replan
  - 宣告整任务完成

### B. 子任务成功判定

当前子任务是否完成，不完全依赖 actor 自报。

runner 会调用 `_apply_subtask_success_override(...)`，结合 observation 规则去做第二层判定。例如：

- `open the phone app`
- `navigate to the contacts section`
- `reach the contact creation entry point`
- contact form fill
- 某些 text-entry 子任务

同时 runner 会阻止“在不可靠 observation 上直接判完成”，例如：

- `observation_consistency == unstable`
- 有明显 `observation_warning`
- `non_system_ui_count == 0`

### C. Contacts 场景特化保护

当前为把 `ContactsAddContact` 跑通，`task_runner.py` 中引入了较多 contact-specific 逻辑，这些逻辑主要集中在：

- `_canonicalize_contact_subtask(...)`
- `_build_contact_form_context(...)`
- `_postprocess_contact_form_subtask(...)`
- `_match_contact_form_fill_success(...)`
- `_extract_contact_identity_from_observation(...)`

其核心目的包括：

- 把 planner 退化出来的单字段目标归一成 grouped form fill
- 检查 actor 是否填错字段、填了不相关字段
- save 后 refresh observation
- 对照 AndroidWorld evaluator 再做一层确认
- 区分：
  - `saved_but_task_check_failed`
  - `saved_with_wrong_identity`
  - `field_misgrounded`

### D. History / Memory / 最终 evaluator

runner 会持续记录：

- actor history
- planner feedback history
- observation warning history
- subtask summary history

同时会把结构化 `MemoryEvent` 写给 `memory_provider.record(...)`。

整任务最终是否成功，不由 planner 或 actor 决定，而是由：

`task.is_successful(env)`

也就是说，当前真实判定链条是：

`actor 自报 -> runner 规则审核 -> AndroidWorld evaluator`

## 4. 当前数据流流程

下面是当前主流程的端到端数据流。

### Step 1. 任务初始化

`task_loop_smoke.py` 负责：

- 连接模型服务
- 连接 AndroidWorld 环境
- load task
- 构造 planner / actor / runner

### Step 2. Observation 采集

`AndroidWorldObservationAdapter.capture_observation(...)`

输出统一 observation dict，作为 planner 的输入。

### Step 3. Planner 生成 subtasks

`AndroidTaskPlanner.plan(...)`

输入：

- 总任务
- 当前 observation
- task history
- memory context

输出：

- `complete_goal`
- 或 `set_tasks`

### Step 4. Runner 规范化 planner subtasks

`task_runner` 会做：

- 格式校验
- contradictory subtask 检查
- contact subtask canonicalization
- dedup
- grounding veto

如果 planner 引用了当前 observation 中不存在的高风险 UI target，runner 会直接 veto 并进入 replan。

### Step 5. Actor 执行单个 subtask

`AndroidActor.run_subtask(...)`

输入：

- 当前 subtask
- 当前 observation
- action history
- memory context

输出：

- 多步 action trace
- 最终 actor status
- final observation

### Step 6. Runner 审核 subtask 结果

runner 会依次做：

1. unstable observation recovery
2. success override
3. contact form postprocess
4. actor failure classification
5. history 写回
6. memory event 写回
7. 判断是否 replan

### Step 7. 整任务完成判定

如果 planner 声称 `complete_goal`，runner 不会直接相信，而是调用：

`task.is_successful(env)`

只有 evaluator 返回成功，整个任务才算真正完成。

### Step 8. Smoke artifacts 落盘

`task_loop_smoke.py` 会将每一轮、每一步的中间结果落盘到：

- `task_loop_smoke_runs/<task_timestamp>/`

常见 artifacts 包括：

- `run_summary.md`
- `run_result.json`
- `round_xx/round_summary.md`
- `planner_prompt.txt`
- `planner_raw_response.txt`
- `planner_result.json`
- `subtask_summary.md`
- `actor_step_xx_decision.md`
- `observation_before.json`
- UI index table

这些 artifacts 是当前项目排查问题最重要的依据。

## 5. 当前完成情况

## 5.1 已完成部分

当前可以明确认为已经完成或基本完成的部分有：

- AndroidWorld 环境接线
- observation 标准化与 screenshot/UI index artifact 生成
- planner 闭环接线
- actor 闭环接线
- task_runner 主闭环
- memory provider 接口预留
- 独立 smoke 评测脚本
- 单元测试与 mock 测试基础设施

换句话说，项目已经具备一个“能真实跑任务、能留痕、能调试”的工程骨架。

## 5.2 已有一定效果的部分

在任务效果上，当前最有进展的是：

- `ContactsAddContact`

这个任务已经不是“完全跑不通”的状态，而是经历过多轮稳定化后，能在真实 AndroidWorld 环境中反复出现成功样本。

这里最关键的原因不是 prompt 单独变强了，而是：

- runner 增加了很多运行时保护
- grouped form fill 有了更明确的后处理
- 完成判定不再只靠 actor 自报

## 5.3 尚未完成部分

以下部分目前仍然没有形成稳定实现：

- 静态记忆系统
- DMS 记忆系统
- 通用 verifier 审核层
- 更系统的 batch evaluation / benchmark 汇总
- 多任务泛化稳定性

特别说明：

- `dms_reproduction/memory/` 当前只有接口，没有实际记忆算法
- `dms_reproduction/verifier/` 当前源码树中没有稳定的 `.py` 实现文件，因此 verifier 仍不能视为正式已落地模块

## 6. 当前主要问题

## 6.1 系统仍然高度依赖 Contacts 场景特化

虽然默认 prompt 已经向通用化方向收敛，但 `task_runner.py` 中为了跑通 `ContactsAddContact`，引入了明显的 contact-specific runtime logic。

这带来的结果是：

- `ContactsAddContact` 受益明显
- 其它任务未必受益
- 泛化性仍然偏弱

## 6.2 prompt 泛化与运行时特化之间还没有稳定平衡

项目最近一段时间的一个核心矛盾是：

- 如果 prompt 太特化，容易只对一个任务有效
- 如果 prompt 完全泛化，当前系统又容易失去稳定性

因此当前系统并不是“只要换 prompt 就能解决问题”，而是 prompt、runner 规则、observation 质量三者共同决定性能。

## 6.3 真正的 verifier 还没有稳定纳入主链路

从项目讨论过程看，verifier 是一个明确的下一阶段方向：

- actor 自报后做第二层审核
- 再接 AndroidWorld evaluator 做第三层最终真值

但从当前源码树看，这一层还没有稳定保留为正式模块。因此现在的完成判定仍主要依赖：

- actor 自报
- runner 内置规则
- AndroidWorld evaluator

## 6.4 多任务泛化能力仍不足

从已有 smoke 经验看，当前系统最有效的任务仍集中在联系人创建路径附近。

而对于：

- draft contact
- settings navigation
- 其他非 contacts 类任务

系统还没有形成同等稳定的运行时机制。

## 6.5 文档状态与代码状态并不完全同步

当前仓库有两个明显问题：

- `docs/architecture_overview.md` 存在编码损坏
- 一些阶段性讨论中的方案，并没有稳定保留在当前源码树中

因此后续对外同步时，应优先以：

- 当前源码
- `tests/`
- `task_loop_smoke_runs/`

作为事实依据，而不是只依赖旧文档。

## 7. 对外讨论时建议强调的结论

如果后续需要和其他同学同步项目进度，我建议直接用下面这几条作为结论：

- 这个 Android GUI PA agent 系统已经完成了从 AndroidWorld observation 到 planner、actor、runner、evaluator 的主闭环实现。
- 当前系统不是 scaffold，而是一个可真实跑 smoke、可留完整 artifacts、可做调试分析的原型系统。
- 当前最稳定的能力集中在 `ContactsAddContact`，其稳定性主要来自 runner 层的运行时保护，而不是记忆系统。
- memory 目前只有接口，没有静态记忆或 DMS 实现。
- verifier 目前还不是稳定落地模块。
- 当前最主要的问题不是“系统能不能跑”，而是“如何在不继续堆任务特化的前提下，把闭环能力泛化到更多任务”。

## 8. 下一阶段建议

结合当前项目状态，下一阶段更合理的推进顺序是：

1. 先明确 verifier 是否要作为正式主链路模块落地。
2. 将 `ContactsAddContact` 中真正通用的运行时保护抽象出来，逐步减少 contact-specific 逻辑散落在 runner 中。
3. 在 memory 已有接口基础上，优先接入一个最小静态记忆版本，再考虑 DMS。
4. 选定 `1-2` 个非 contacts 任务作为泛化验证对象，而不是继续只在单一任务上打补丁。
5. 修复或重写 `docs/architecture_overview.md`，避免后续协作时继续引用损坏文档。

---

本文档基于当前仓库源码、现有 smoke 脚本、tests 目录结构，以及当前保留的 docs 内容整理，目标是为后续进度同步、方案讨论和模块边界澄清提供一个统一版本。
