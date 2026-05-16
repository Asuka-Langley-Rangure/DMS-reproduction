# Android GUI Agent Handoff

## 当前目标

当前仓库是在 AndroidWorld 环境之上搭建一个可迭代的 Android GUI agent 闭环。今天已经打通的主线是：

- `planner -> android_actor -> task_runner -> task_loop_smoke.py`
- 目标是先稳定完成单任务闭环，再逐步接入记忆系统
- memory 目录与设计仍在，但还没有真正接入主运行时写回

当前最重要的不是继续扩展能力，而是让现有闭环在真实 smoke 中稳定推进，而不是卡在 observation 退化、planner 低层目标回退或 grounding veto 停滞。

## 当前实现状态

### Planner

位置：
- [dms_reproduction/agents/planner.py](/f:/baoyantest/dms/dms_reproduction/agents/planner.py)

当前接口：
- 输入：
  - `user_goal: str`
  - `observation: dict`
  - `task_history: list[dict]`
  - `memory_context: str`
- 输出：
  - `PlannerResult`

关键数据结构：
- `PlannerSubtask`
  - `precondition`
  - `goal`
  - `reason`
  - `agent`
- `PlannerResult`
  - `is_goal_complete`
  - `completion_message`
  - `subtasks`
  - `raw_response`
  - `parse_error`
  - `repaired_parse`
  - `repair_reason`

当前能力：
- 仍使用 `Precondition: ... Goal: ...` task string
- prompt 构造已稳定
- parser 兼容：
  - `set_tasks`
  - `set_tasks_with_agents`
  - `tasks`
  - `task_assignments`
- 已有 near-JSON repair
- 已支持 history / memory 注入
- 已在 prompt 中补了联系人表单 grouped subtask 倾向：
  - 优先 `Fill in ... in the contact form.`
  - 不鼓励默认拆成单字段目标

当前限制：
- planner 仍主要靠 prompt 收敛，不靠复杂 validator
- parser 可以修轻量结构错，但不负责高层 planning 质量
- 目前仍会在某些轮次退化成低层 action-style 目标，例如 `Tap the 'Create new contact' button.`

### Android Actor

位置：
- [dms_reproduction/agents/android_actor.py](/f:/baoyantest/dms/dms_reproduction/agents/android_actor.py)

当前接口：
- 输入：
  - `ActorRequest`
    - `subtask`
    - `observation`
    - `action_history`
    - `memory_context`
- 输出：
  - `ActorRunResult`

关键数据结构：
- `ActorAction` 及其子类
  - `status`
  - `answer`
  - `click`
  - `long_press`
  - `input_text`
  - `keyboard_enter`
  - `navigate_home`
  - `navigate_back`
  - `scroll`
  - `open_app`
  - `wait`
- `ActorStepResult`
- `ActorRunResult`

当前能力：
- ActorCode 风格 prompt
- JSON action parsing
- action alias normalization
  - `type -> input_text`
  - `enter_text -> input_text`
  - `fill_text -> input_text`
  - `set_text -> input_text`
- clickable / editable guard
- 轻量 unique-candidate correction
- degraded / unstable observation 下的约束
- history / memory 注入
- 组表单目标的轻量提示：
  - 若 subtask 是 coherent form section，只填对应字段并停止

当前限制：
- actor 仍不是通用强 agent，仍包含联系人场景导向的局部规则
- 还没有一套通用的 task-facts 绑定层
- 当前 grouped form fill 更多是 prompt + runner 支持，不是完整 actor scope engine

### Task Runner

位置：
- [dms_reproduction/agents/task_runner.py](/f:/baoyantest/dms/dms_reproduction/agents/task_runner.py)

当前接口：
- `AndroidTaskRunner.run_task(env, task, user_goal) -> TaskRunResult`

关键数据结构：
- `TaskRunConfig`
- `TaskRunResult`
- `PlannerRoundRecord`
- `SubtaskRunRecord`

当前能力：
- planner-actor 最小闭环
- planner round 记录
- actor step 累积与 history 追加
- post-action success override
- unstable observation recovery
- planner grounding veto
- grouped contact-form progress 记录
- round limit / actor error / planner error 统一收束

当前 grouped form 逻辑：
- 组目标示例：
  - `Fill in Mia Garcia and +18856139998 in the contact form.`
- 当前 runner 能识别组目标并检查：
  - `first_name`
  - `last_name`
  - `phone`
- 可记录：
  - `completed_fields`
  - `remaining_fields`
  - `actual_values`
- 可区分：
  - 完整完成
  - `progress_made_but_not_complete`
  - 字段级 fallback 仍保留，但应作为次级路径

当前限制：
- grouped form fill 目前只在 runner success detection 层支持
- 不代表 planner 已稳定生成这类 goal
- 更不代表系统已经稳定进入 contact editor 阶段

### Smoke

位置：
- [scripts/task_loop_smoke.py](/f:/baoyantest/dms/scripts/task_loop_smoke.py)

当前能力：
- 真实环境 planner + actor 闭环 smoke
- 每轮 artifact 落盘
- 图片、prompt、raw response、decision summary、UI index table 全保留
- 输出：
  - `run_summary.md`
  - `run_result.json`
  - `artifact_index.json`
  - 每轮 / 每 subtask / 每 step 明细文件
- 新增联系人表单 artifact：
  - `form_fill_progress.json`

## 关键接口与数据流

### Observation contract

统一通过 `AndroidWorldObservationAdapter.capture_observation(...)` 生成 observation dict。

当前常见字段包括：
- `goal`
- `current_activity`
- `foreground_package`
- `app_name`
- `screen_size`
- `ui_elements`
- `ui_description`
- `valid_ui_indices`
- `visible_ui_count`
- `clickable_ui_count`
- `non_system_ui_count`
- `editable_ui_count`
- `keyboard_active_context`
- `observation_warning`
- `observation_consistency`
- `screenshot_b64`
- `labeled_screenshot_b64`

### 一轮闭环数据流

1. `task_runner` 调 `capture_observation`
2. `planner.build_messages(...)`
3. `planner.plan(...)`
4. `planner.parse_response(...)`
5. `task_runner` 对 planner 结果做：
   - normalization
   - grounding check
6. `actor.run_subtask(...)`
7. `task_runner` 对 actor 结果做：
   - success override
   - unstable observation recovery
   - history 聚合
   - replan 决策
8. `task_loop_smoke.py` 将全过程写入 artifact

### Planner-facing history 与 actor-facing history

当前实现已经开始区分这两类用途：

- actor-facing history
  - 保留 step 级细节
  - 用于避免重复动作和继续当前 subtask
- planner-facing history
  - 已开始做摘要化
  - 当前特别是联系人 grouped form 场景，会追加 `subtask_summary`
  - 目标是让 planner 看到 milestone，而不是每一次 click / input

## 当前测试覆盖

### Planner tests

位置：
- [tests/test_planner.py](/f:/baoyantest/dms/tests/test_planner.py)

当前覆盖：
- prompt 必要段落
- `Precondition: ... Goal: ...` 要求
- near-JSON repair
- parser 兼容多种 tool / task 字段形式
- 联系人 grouped form prompt 倾向

### Android actor tests

位置：
- [tests/test_android_actor.py](/f:/baoyantest/dms/tests/test_android_actor.py)

当前覆盖：
- prompt 结构
- observation warning 注入
- action parsing
- alias normalization
- clickable / editable guard
- degraded observation complete 限制
- unique target correction
- execution loop
- parse / execution / step_limit 路径

### Task runner tests

位置：
- [tests/test_task_runner.py](/f:/baoyantest/dms/tests/test_task_runner.py)

当前覆盖：
- planner completed / parse_error
- 多 subtask 顺序执行
- actor failure replan
- success override
- unstable observation recovery
- planner grounding veto
- grouped contact-form full success
- grouped contact-form partial progress

### Smoke artifact tests

位置：
- [tests/test_task_loop_smoke_artifacts.py](/f:/baoyantest/dms/tests/test_task_loop_smoke_artifacts.py)

当前覆盖：
- 图片落盘
- round / subtask summary
- actor seen image
- 轻量 `run_result.json`
- `form_fill_progress.json`

### 测试局限

- 绝大多数测试是 mock-based，不是 emulator 真实交互
- grouped form fill 逻辑虽然有单测，但在真实 smoke 中还没有真正命中过
- 当前真实 smoke 仍主要卡在进入 contact creation flow 之前

## 最新真实日志结论

基于：
- [task_loop_smoke_runs/ContactsAddContact_20260515_220450/run_summary.md](/f:/baoyantest/dms/task_loop_smoke_runs/ContactsAddContact_20260515_220450/run_summary.md)

最新 run 的主结论：

1. **round 1 的 observation 仍然不稳定**
   - planner 给出 `Open the Phone app.`
   - actor / runner 通过 success override 把它判定为完成
   - 但 round 1 结束原因仍是 `observation_unstable_persisted`
   - final observation 基本只剩 `system UI`

2. **round 2+ planner 会退化成低层 action-style goal**
   - 例子来自：
     - [round_02/planner_result.json](/f:/baoyantest/dms/task_loop_smoke_runs/ContactsAddContact_20260515_220450/round_02/planner_result.json)
     - `Goal: Tap the 'Create new contact' button.`
   - 这是 planning granularity 问题，不是 parse 问题

3. **planner grounding veto 在工作，但把系统卡住了**
   - round 2-5 全部被 `planner_subtask_not_grounded_in_observation` 拦住
   - 证据在：
     - [round_02/planner_grounding_check.json](/f:/baoyantest/dms/task_loop_smoke_runs/ContactsAddContact_20260515_220450/round_02/planner_grounding_check.json)
   - 说明系统已经不会盲目继续执行坏 subtask
   - 但 planner 没有从失败反馈恢复成更高层、更合理的 functional goal

4. **今天新增的 grouped form fill 逻辑还没有真正命中**
   - `group_form_subtask_used = 0`
   - `form_fill_partial_progress = 0`
   - `field_level_fallback_used = 0`
   - 说明当前主阻塞仍在 contact editor 之前
   - 不能把 grouped form fill 当作当前主故障点

5. **actor overshoot 已经不是当前主阻塞**
   - `actor_overshoot_after_goal = 1`
   - 表示 success override 仍然发生过
   - 但当前真正卡死闭环的是：
     - unstable observation
     - planner 低层 goal 回退
     - grounding veto 后无恢复

## 当前已知问题与优先级

### P0: success override 后 observation 仍不稳定

现象：
- `Open the Phone app` 被判定成功
- 但 final observation 只剩 `system UI`
- round 1 最终仍 `observation_unstable_persisted`

根因判断：
- success override 与 observation stabilization 之间仍然脱钩
- 可能是 activity / screenshot / a11y tree 采样时序不一致

证据：
- [round_01/subtask_01/subtask_summary.md](/f:/baoyantest/dms/task_loop_smoke_runs/ContactsAddContact_20260515_220450/round_01/subtask_01/subtask_summary.md)
- [round_01/observation_consistency_report.json](/f:/baoyantest/dms/task_loop_smoke_runs/ContactsAddContact_20260515_220450/round_01/observation_consistency_report.json)

推荐修法：
- 先查 success override 之后的 observation capture 时序
- 重点检查：
  - activity 已切换但 screenshot/UI tree 未跟上
  - local recovery 为什么仍不能恢复稳定 observation

### P1: planner 在 contact-creation 入口处退化成 atomic goal

现象：
- `Navigate to contacts` 之后，planner 回退成：
  - `Tap the 'Create new contact' button.`

根因判断：
- 当前 prompt 仍不足以稳定阻止 atomic goal 回退
- 尤其在 observation 已退化时，planner 容易输出低层动作式目标

证据：
- [round_02/planner_result.json](/f:/baoyantest/dms/task_loop_smoke_runs/ContactsAddContact_20260515_220450/round_02/planner_result.json)

推荐修法：
- 不继续大改 prompt 结构
- 重点改“失败反馈如何写回 planner”
- 让 planner 在 grounding veto 后更倾向返回：
  - `Reach the contact creation entry point.`
  - 或类似更高层 functional goal
- 而不是直接指定 tap 某按钮

### P1: grounding veto 缺少恢复策略

现象：
- round 2-5 一直是 `planner_subtask_not_grounded_in_observation`

根因判断：
- veto 机制本身有效
- 但 veto 后系统只会继续 replan，没有更强的 recovery path

证据：
- [run_summary.md](/f:/baoyantest/dms/task_loop_smoke_runs/ContactsAddContact_20260515_220450/run_summary.md)

推荐修法：
- 在 grounding veto 后，不只是把 reason 塞回 planner
- 还应结合当前 observation 状态给 planner 更明确的恢复提示
- 或让 runner 在特定入口阶段优先请求 observation refresh / local navigation recovery

### P2: grouped contact-form fill 仍未在真实 smoke 命中

现象：
- 相关计数全为 0

根因判断：
- grouped form fill 逻辑实现了，但当前执行尚未推进到 contact editor

推荐修法：
- 暂时不要优先继续调 grouped form fill
- 等入口阶段稳定后，再验证：
  - group form success
  - partial progress
  - field fallback

## 今天做过什么、效果如何、哪些要谨慎

### 初始状态

- planner 和 observation adapter 基本可用
- actor 还比较薄
- 没有稳定的 planner-actor 闭环

### 今天已完成的搭建

- actor JSON action schema
- actor prompt / parsing / alias normalization
- task_runner 闭环
- smoke artifact 体系
- observation consistency / unstable detection
- success override
- planner parse repair
- planner grounding veto
- grouped contact-form handling

### 已观察到的效果

- 比最初稳很多，不再只是裸 parse 和裸执行
- 现在已经能：
  - 明确看到 prompt / raw response / images
  - 对坏 planner subtask 做 veto
  - 对部分 actor 错误做本地恢复

### 当前要谨慎的地方

- 有些修复会引入更强 veto，导致系统进入“不会乱做，但也过不去”的状态
- grouped form fill 的实现不能说明联系人创建已经稳定
- 不建议再大改 planner prompt 主结构
- 不建议继续堆 contact-specific validator
- 不建议把当前主阻塞误判成字段填写细节

## 建议下一步

1. **先修 round 1 之后的 observation 质量问题**
   - 检查 success override 之后的 observation capture 时序
   - 对比：
     - activity
     - screenshot
     - accessibility tree
   - 查 local recovery 为什么不能把 `system-ui-only` 拉回稳定页面

2. **再修 planner 在入口阶段的 functional goal 稳定性**
   - 重点不是字段怎么填
   - 而是它为什么从 `Navigate to contacts` 退化成 `Tap create new contact`
   - 让 planner 在 grounding veto 后更偏向高层 milestone，而不是按钮级动作

3. **最后再验证 grouped form fill 路径**
   - 等 contact editor 能稳定进入后
   - 再看：
     - grouped form success
     - partial progress
     - field fallback

## 相关入口文件

- [dms_reproduction/agents/planner.py](/f:/baoyantest/dms/dms_reproduction/agents/planner.py)
- [dms_reproduction/agents/android_actor.py](/f:/baoyantest/dms/dms_reproduction/agents/android_actor.py)
- [dms_reproduction/agents/task_runner.py](/f:/baoyantest/dms/dms_reproduction/agents/task_runner.py)
- [scripts/task_loop_smoke.py](/f:/baoyantest/dms/scripts/task_loop_smoke.py)
- [tests/test_planner.py](/f:/baoyantest/dms/tests/test_planner.py)
- [tests/test_android_actor.py](/f:/baoyantest/dms/tests/test_android_actor.py)
- [tests/test_task_runner.py](/f:/baoyantest/dms/tests/test_task_runner.py)
- [tests/test_task_loop_smoke_artifacts.py](/f:/baoyantest/dms/tests/test_task_loop_smoke_artifacts.py)
