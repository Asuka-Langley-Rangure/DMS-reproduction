# 2026-05-17 项目进展与交接说明

## 1. 今日目标与总体结论

今天的工作重点不是继续堆 task-specific 逻辑，而是把系统从“依赖 runner 特化纠偏”逐步拉回到“planner/verifier/runner 各自职责清晰”的形态，并把当前真实瓶颈暴露出来。

今天完成后的整体结论是：

- `first-pass stage_plan` 已经基本成型，能先验地对任务做 3-5 个高层 milestone 分解。
- `second-pass planner` 仍然是当前主要瓶颈，但它的问题已经从“系统自己把结果改坏”收敛到了“planner 本体输出质量不足”。
- `complete_goal` 现在已经降级为 `completion candidate`，只有 AndroidWorld evaluator 通过才会真正结束任务。
- `second-pass current_stage_id` 的系统保真问题已经修掉，后续可以更可信地分析 planner 自身的 stage 选择问题。

## 2. 今天新增/修改的核心能力

### 2.1 新增 LLM verifier，并接入主链路

新增了 `dms_reproduction/agents/verifier.py`，输出结构固定为：

```json
{
  "status": "success | failure | uncertain",
  "reason": "...",
  "memory_eligible": true
}
```

当前 verifier 的角色：

- 基于 `subtask + action history + before/after observation`
- 优先使用 actor 声称完成时对应的 observation 作为主证据
- runner 不再直接相信 actor 的 `completed`
- `memory_eligible` 当前只记录，不参与任务完成判定

当前主链路已经变成：

`planner -> actor -> verifier -> runner -> AndroidWorld evaluator`

### 2.2 修复 observation screenshot/tree 不一致问题

之前出现过：

- screenshot 已经切到 `Contacts`
- 但 UI tree 还是 `Voicemail`
- 最终导致 labeled screenshot 框标错

今天已经在 observation pipeline 中补了更强的一致性检查与重采样逻辑：

- 对连续 stable capture 做一致性确认
- 如果 screenshot/tree 不一致，继续重采样
- 不再轻易把这种 stale tree 标成 stable

这一块在真实 smoke 中已经明显压住了早先那类错框问题。

### 2.3 完成 baseline_before_refactor

已生成：

- [baseline_before_refactor.md](/f:/baoyantest/dms/baseline_before_refactor.md)

覆盖了 3 个任务各 3 次运行，用作后续重构对比基准：

- `ContactsAddContact`
- `ContactsNewContactDraft`
- `SystemWifiTurnOnVerify`

记录字段包括：

- 任务是否成功
- 失败在哪个 round
- planner 输出了什么
- actor 最后做了什么
- runner 为什么判成功/失败
- AndroidWorld evaluator 结果

### 2.4 去掉 runner 中 contact-specific planner shaping

今天明确做了一个方向性调整：

- 移除了 runner 中专门“改 planner 输出形态”的 contact-specific 逻辑
- 不再自动把字段级 subtask 改成 grouped form fill
- 不再自动扩 canonical stage plan
- 不再依赖 runner 强行把 planner_result 改成看起来更合理

保留了执行后 contact-specific 真值保护逻辑，例如：

- `saved_with_wrong_identity`
- `field_misgrounded`
- `saved_but_task_check_failed`
- `saved_contact_state_changed`

这意味着：

- 现在更容易暴露 planner 本体真实能力
- 成功率短期可能下降
- 但更适合做 prompt 和架构层面的真实优化

### 2.5 planner 改成 two-pass 结构

今天把 planner 从“单次调用同时做全局分解和当前步规划”改成了明确的 two-pass：

#### first-pass

`plan_stage_milestones(user_goal)`

- 只看 `user_goal`
- 不看 `observation`
- 输出 3-5 个 whole-task milestones

#### second-pass

`plan_current_subtasks(user_goal, stage_plan, observation, task_history, memory_context)`

- 使用 frozen `stage_plan`
- 再结合当前环境决定：
  - 当前处于哪个 stage
  - 本轮输出 1-2 个 subtasks

新增 artifacts：

- `initial_stage_plan.json`
- `initial_stage_plan_messages.json`
- `initial_stage_plan_prompt.txt`
- `initial_stage_plan_raw_response.txt`

### 2.6 complete_goal 语义已重构

当前 planner 仍然保留 `complete_goal` 的权力，但 runner 语义已经改成：

- `complete_goal` 只是 completion candidate
- 只有 `task.is_successful(env)` 返回成功，任务才真正 `completed`
- 如果 evaluator 拒绝：
  - 写 `planner_complete_but_task_check_failed` 到 history
  - 当作一次 planner failure
  - 下一轮必须继续规划补救/验证子任务
- 并且这类失败不再触发 stage-plan 重建

### 2.7 second-pass state fidelity 已修复

这是今天最重要的低层修复之一。

之前存在一个系统保真 bug：

- second-pass raw output 明明返回了 `current_stage_id=2/3`
- 但 parser/runner 会把它吞掉
- 最终 artifacts 里经常落成 `current_stage_id=1`

现在已修复为：

- second-pass 即使不返回 `stage_plan`，只返回 `current_stage_id + tasks`
- `current_stage_id` 也会被保留下来
- runner 不再静默把非法或缺失 stage 回退成 stage 1
- 非法 stage 现在会显式记成 `planner_current_stage_invalid`

同时 strict parse 继续保留，并给 planner parse error 增加了显式错误分类：

- `planner_task_format_invalid`
- `planner_stage_plan_invalid`
- `planner_json_parse_failed`
- `planner_tool_unsupported`
- 等

现在 `planner_result.json` 和 round summary 已经能直接看出：

- 是格式失败
- 还是 stage plan 失败
- 还是普通 JSON 失败

## 3. 今天的真实 smoke 观察

### 3.1 FilesDeleteFile

在 `FilesDeleteFile` 上，today 的关键发现不是 first-pass，而是 second-pass：

- first-pass `stage_plan` 已经基本符合“先验 whole-task milestones”的定义
- 但 second-pass 之前会在 launcher/home screen 直接 premature `complete_goal`

在 `complete_goal candidate` 语义重构之后，这类 premature completion：

- 不再直接结束任务
- 会被 evaluator 拒绝
- 然后写回 `planner_complete_but_task_check_failed`

这让系统行为更安全，也更接近我们想要的 evaluator-centered truth model。

### 3.2 ContactsAddContact

最新真实 run：

- [ContactsAddContact_20260517_234540](/f:/baoyantest/dms/task_loop_smoke_runs/ContactsAddContact_20260517_234540)

这个 run 的意义很重要：

- 它证明了 second-pass `current_stage_id` 保真已经修好
- 例如 round 3/4/5 中，raw `current_stage_id=2`，最终 `planner_result.current_stage_id` 也保持为 `2`
- 之前“raw 是 2/3，最后被系统写回 1”的问题已经消失

这个 run 也同时暴露了当前真正剩下的 planner 问题：

- planner 已经能选对 stage
- 但 subtask wording 仍会退化成 button-click 级别
- 例如：
  - `Click the New Contact button to navigate to the New Contact screen.`
- 最终卡在：
  - `same_subtask_no_progress`

也就是说，当前已经可以更明确地说：

- 现在的主要瓶颈不再是系统保真
- 而是 second-pass prompt 对 subtask 粒度和推进能力约束还不够

## 4. 当前仍存在的问题

### 4.1 second-pass subtask 粒度仍偏低

虽然 stage 选择现在更可信了，但 planner 仍然经常把 stage 2 的 subtask 写成：

- `Click ...`
- `Tap ...`
- `Input ...`

而不是更高层的 milestone wording。

### 4.2 same_subtask_no_progress 仍然常见

尤其在联系人任务中，planner 经常：

- 连续多轮重复同一 stage 的同一类 subtask
- 不会在失败后切换为更保守的修复 milestone
- 也不会自然推进到下一个更合理阶段

### 4.3 strict parse failure 仍在暴露模型格式问题

现在 parser 是 strict 的，仍然经常抓到：

- 少 `Precondition:`
- `Goal:` 结构不合法
- near-JSON 输出

这本身不是 bug，而是我们刻意保留下来的“真实暴露 planner 输出质量”的手段。

## 5. 下一个跟进者建议优先做什么

建议下一位接手时，优先级如下：

### 第一优先级：继续优化 second-pass prompt

目标不是再改 first-pass，而是收 second-pass：

- 避免 gesture-shaped subtasks
- 要求 stage 内 subtask 使用 milestone wording
- 避免 `same_subtask_no_progress`
- 在 repeated failure 后，切换为 repair / verification / safer-progress milestone

重点关注：

- stage 2 不要反复输出 `Click the New Contact button...`
- 应该输出更高层的“到达 New Contact screen”式目标

### 第二优先级：观察 planner_task_format_invalid 的出现频率

现在格式错误已经有显式标签了，可以统计：

- 是不是 second-pass 普遍还有 `Precondition` 漏写
- 是不是只是某类任务里更高发

这会决定后续是继续 strict，还是未来要做有限的结构修复。

### 第三优先级：继续用真实 smoke，而不是只看 unit test

目前单测已经能保证：

- verifier 接口行为
- two-pass planner 主接口
- `complete_goal candidate` 控制流
- second-pass `current_stage_id` 保真

但真正影响效果的仍然是 smoke 中的：

- planner wording
- stage 推进
- repeated no-progress

## 6. 本地运行方式

当前可用的真实 smoke 运行方式是直接使用指定解释器，不依赖当前 shell 的 `conda activate`：

```powershell
$env:PYTHONPATH = "f:\baoyantest\dms\android_world"
& "F:\.conda\envs\android_world\python.exe" scripts\task_loop_smoke.py --skip_emulator_launch
```

如果要指定任务，例如：

```powershell
$env:PYTHONPATH = "f:\baoyantest\dms\android_world"
& "F:\.conda\envs\android_world\python.exe" scripts\task_loop_smoke.py --skip_emulator_launch --task ContactsAddContact
```

## 7. 本次交接的关键信息

如果只保留三句话给下一位同学：

1. `first-pass stage_plan` 现在已经不是主问题，真正的问题在 `second-pass`。  
2. `current_stage_id` 被系统吞掉的问题今天已经修好，现在可以更可信地分析 planner 自身问题。  
3. 下一步不要再往 runner 里堆 task-specific 纠偏，而应该优先优化 second-pass prompt 的 subtask 粒度和推进能力。  
