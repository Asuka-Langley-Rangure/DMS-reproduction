# Baseline Before Refactor

基线采集日期：`2026-05-17`

本文件记录重构前的真实闭环 baseline。目标是给后续“重构是否变好”提供可对照的前置样本，而不是只看单次成功截图。

本次最终采用的 3 个任务：

- `ContactsAddContact`
- `ContactsNewContactDraft`
- `SystemWifiTurnOnVerify`

未纳入本轮 baseline 的任务：

- `SimpleCalendarAddOneEvent`：任务初始化时报错 `/data/data/com.simplemobiletools.calendar.pro/databases does not exist`，属于当前环境缺件，不适合作为 agent 基线。
- `SystemBrightnessMaxVerify`：预跑时 observation 阶段直接失败，报 `Could not get a11y tree.`，本轮不纳入正式三任务对比。

## Aggregate

| Task | Success / 3 | Main failure mode |
| --- | --- | --- |
| `ContactsAddContact` | `1 / 3` | post-save control flow and task check mismatch |
| `ContactsNewContactDraft` | `0 / 3` | planner grounding veto or invalid actor index |
| `SystemWifiTurnOnVerify` | `0 / 3` | actor repeatedly outputs invalid click indices |

## ContactsAddContact

### Run 1

- Artifacts: [ContactsAddContact_20260517_161050](/f:/baoyantest/dms/task_loop_smoke_runs/ContactsAddContact_20260517_161050)
- Success: `No`
- Failed at round: `5`
- AndroidWorld evaluator: `None`
- Planner output:
  - `R1` Open the Phone app.
  - `R2` Click on the `Contacts` tab to view the contact list.
  - `R3` Click on the `Create new contact` button to start creating a new contact.
  - `R4` Enter the first name `William`.
  - `R5` Fill in `William dos Santos` and `+15099873339` in the contact form.
- Actor last action:
  - Round `5` last step returned `{"action_type":"status","goal_status":"complete","message":"The contact William dos Santos has been successfully created and saved."}`
- Runner decision:
  - 最后一次 replan reason 是 `subtasks_exhausted`
  - actor 自报完成，但 runner 没有把整任务判成成功，最终也没有拿到 evaluator `True`

### Run 2

- Artifacts: [ContactsAddContact_20260517_161246](/f:/baoyantest/dms/task_loop_smoke_runs/ContactsAddContact_20260517_161246)
- Success: `Yes`
- Failed at round: `N/A`
- AndroidWorld evaluator: `True`
- Planner output:
  - `R1` Open the Phone app.
  - `R2` Tap on `Create new contact` to start adding Noah Muller's details.
  - `R3` Enter `Noah` into the First name field.
  - `R4` Enter `Noah` into the First name field.
  - `R5` `complete_goal` with message `The contact Noah Muller has been successfully created and saved.`
- Actor last action:
  - 最后一个实际 actor round 是 `4`
  - 最后一步返回 `{"action_type":"status","goal_status":"complete","message":"Contact Noah Muller has been created and saved."}`
- Runner decision:
  - runner 最终接受保存后的状态变化，terminal failure marker 里能看到 `saved_contact_state_changed`
  - 随后 planner 在 `R5` 直接 `complete_goal`
  - AndroidWorld evaluator 返回 `True`，所以整任务判成功

### Run 3

- Artifacts: [ContactsAddContact_20260517_161433](/f:/baoyantest/dms/task_loop_smoke_runs/ContactsAddContact_20260517_161433)
- Success: `No`
- Failed at round: `5`
- AndroidWorld evaluator: `None`
- Planner output:
  - `R1` Open the Phone app.
  - `R2` Tap on the `Create new contact` button to start creating a new contact.
  - `R3` Input first name `Alice`; input last name `Alves`; enter phone `+14135002526`; click `Save`.
  - `R4` `complete_goal`: `The contact 'Alice Alves' has been successfully created with the number +14135002526.`
  - `R5` `complete_goal`: `The contact 'Alice Alves' has been successfully created.`
- Actor last action:
  - 最后一个实际 actor round 是 `3`
  - 最后一步返回 `{"action_type":"status","goal_status":"complete","message":"The contact 'Alice Alves' has been successfully created."}`
- Runner decision:
  - 最后一次 replan reason 是 `planner_complete_but_task_check_failed`
  - terminal failure marker 是 `saved_with_wrong_identity`
  - 也就是说 planner 和 actor 都以为完成了，但 runner/后验检查不接受这次保存结果

## ContactsNewContactDraft

### Run 1

- Artifacts: [ContactsNewContactDraft_20260517_161809](/f:/baoyantest/dms/task_loop_smoke_runs/ContactsNewContactDraft_20260517_161809)
- Success: `No`
- Failed at round: `5`
- AndroidWorld evaluator: `None`
- Planner output:
  - `R1` Open the Contacts app.
  - `R2-R5` Click the `+` button to add a new contact.
- Actor last action:
  - Round `5` last step输出 `{"action_type":"click","index":100}`
  - actor status 是 `parse_error`
- Runner decision:
  - 最后一次 replan reason 是 `invalid_index_error`
  - 核心问题不是任务理解，而是 actor 持续给出超出当前 observation 的无效 index

### Run 2

- Artifacts: [ContactsNewContactDraft_20260517_161904](/f:/baoyantest/dms/task_loop_smoke_runs/ContactsNewContactDraft_20260517_161904)
- Success: `No`
- Failed at round: `1-5`
- AndroidWorld evaluator: `None`
- Planner output:
  - `R1-R5` 都是 `Open the Contacts app.`
- Actor last action:
  - `None`
  - actor 没有真正执行；runner 在 actor 之前就拦下来了
- Runner decision:
  - 每轮都是 `planner_subtask_not_grounded_in_observation`
  - grounding check 显示 planner 引用了当前 observation 中不存在的 `Contacts` target，所以 runner 直接 veto 并 replan

### Run 3

- Artifacts: [ContactsNewContactDraft_20260517_161933](/f:/baoyantest/dms/task_loop_smoke_runs/ContactsNewContactDraft_20260517_161933)
- Success: `No`
- Failed at round: `1-5`
- AndroidWorld evaluator: `None`
- Planner output:
  - `R1-R5` 都是 `Open the Contacts app.`
- Actor last action:
  - `None`
  - actor 没有执行
- Runner decision:
  - 与上一条相同，连续 `planner_subtask_not_grounded_in_observation`
  - 说明这个任务当前主要卡在 planner grounding，而不是 actor 执行

## SystemWifiTurnOnVerify

### Run 1

- Artifacts: [SystemWifiTurnOnVerify_20260517_161714](/f:/baoyantest/dms/task_loop_smoke_runs/SystemWifiTurnOnVerify_20260517_161714)
- Success: `No`
- Failed at round: `5`
- AndroidWorld evaluator: `None`
- Planner output:
  - `R1` Access the settings menu to turn on Wi-Fi.
  - `R2-R5` Open the Settings app to access the Wi-Fi settings.
- Actor last action:
  - Round `5` last step输出 `{"action_type":"click","index":432}`
  - actor reason 是“Settings app is visible...”
  - actor status 是 `parse_error`
- Runner decision:
  - 最后一次 replan reason 是 `invalid_index_error`
  - 失败模式很稳定：actor 想点对目标，但给了一个明显无效的 index

### Run 2

- Artifacts: [SystemWifiTurnOnVerify_20260517_162002](/f:/baoyantest/dms/task_loop_smoke_runs/SystemWifiTurnOnVerify_20260517_162002)
- Success: `No`
- Failed at round: `5`
- AndroidWorld evaluator: `None`
- Planner output:
  - `R1` Navigate to the settings menu.
  - `R2-R5` Click on the Settings app to proceed towards turning on Wi-Fi.
- Actor last action:
  - Round `5` last step输出 `{"action_type":"click","index":420}`
  - actor status 是 `parse_error`
- Runner decision:
  - 最后一次 replan reason 是 `invalid_index_error`
  - 与 Run 1 同型，问题集中在 actor index grounding

### Run 3

- Artifacts: [SystemWifiTurnOnVerify_20260517_162047](/f:/baoyantest/dms/task_loop_smoke_runs/SystemWifiTurnOnVerify_20260517_162047)
- Success: `No`
- Failed at round: `5`
- AndroidWorld evaluator: `None`
- Planner output:
  - `R1` Access the settings menu to turn on Wi-Fi.
  - `R2-R5` Click on `Network & internet` to proceed towards turning on Wi-Fi.
- Actor last action:
  - Round `5` last step输出 `{"action_type":"click","index":5}`
  - actor reason 是“`Network & internet` option is visible...”
  - actor status 是 `parse_error`
- Runner decision:
  - 最后一次 replan reason 是 `invalid_index_error`
  - 这次目标语义比前两条更接近正确路径，但最后仍然死在 index 校验

## Baseline Read

- `ContactsAddContact` 是当前唯一能在真实 fresh run 里成功完成的任务，但成功率只有 `1/3`，且失败已经从“完全不会做”变成了“保存后判定链不稳定”。
- `ContactsNewContactDraft` 当前有两种失败形态：
  - planner grounding 直接被 runner veto，actor 根本没机会执行
  - actor 执行时反复产出无效 `+` 按钮 index
- `SystemWifiTurnOnVerify` 的失败形态非常一致：
  - planner 大方向通常没错
  - actor 在 launcher 或 settings 页面上持续输出错误 index
  - runner 因 `invalid_index_error` 连续 replan 到 round limit

这份 baseline 可以直接作为后续 refactor 后的对照模板，重点比对：

- 成功率是否提升
- 同类失败是否减少
- grounding veto 是否减少
- invalid index 是否减少
- `ContactsAddContact` 是否从 `1/3` 提升，并且后验检查更稳定
