# Planner 模块文档

## Overview

`planner` 是当前 DMS 复现工程里负责 **task-level planning** 的模块。它接收：

- 用户总目标 `user_goal`
- 当前 Android 界面的标准化 observation
- 当前 session 内的子任务历史 `task_history`
- 可选的长期记忆上下文 `memory_context`

然后返回：

- `complete_goal`：总目标已完成
- `set_tasks`：下一轮 1-5 个功能性子任务

当前实现边界如下：

- 只做 **task-level planning**
- 不输出 atomic action
- 不执行 `android_actor`
- 不更新 memory
- 不做多轮自动 replanning，当前只支持由外部循环重复调用

当前模块与其他组件的关系：

- `AndroidWorldObservationAdapter`
  - 从 AndroidWorld 环境抓取状态并构造成 planner 可消费的 observation
- `OpenAICompatibleClient`
  - 将 planner 生成的 multimodal messages 发给 vLLM / OpenAI-compatible 服务
- `scripts/planner_smoke.py`
  - 负责真实链路 smoke：启动/复用 tunnel、检查模拟器、抓 observation、调 planner、保存结果

## Core Flow

从 `scripts/planner_smoke.py` 出发，一次 planner 调用的完整流程如下：

1. 启动或复用模型 tunnel
2. 检查或启动 Android 模拟器
3. 初始化 AndroidWorld task
4. 使用 `AndroidWorldObservationAdapter.capture_observation(...)` 抓取 observation
5. 构造 `OpenAICompatibleClient`
6. 构造 `AndroidTaskPlanner`
7. 调用 `planner.build_messages(...)` 生成 multimodal messages
8. 调用 `planner.plan(...)`
9. `OpenAICompatibleClient.generate(...)` 向模型发送请求
10. `planner.parse_response(...)` 解析模型返回的 JSON
11. `planner_smoke.py` 将 prompt、raw response、result、meta 等文件落盘

可以把它理解成这条最小链路：

```text
AndroidWorld env
  -> AndroidWorldObservationAdapter
  -> AndroidTaskPlanner.build_messages()
  -> OpenAICompatibleClient.generate()
  -> AndroidTaskPlanner.parse_response()
  -> planner_smoke_runs/<task_timestamp>/
```

## API Reference

### `PlannerSubtask`

定义位置：
- [dms_reproduction/agents/planner.py](/f:/baoyantest/dms/dms_reproduction/agents/planner.py)

字段：

- `precondition: str`
  - 子任务前置条件
- `goal: str`
  - 子任务目标
- `reason: str`
  - 生成该子任务的原因说明
- `agent: Optional[str]`
  - 当前仅作兼容占位，默认会落成 `android_actor`

属性与方法：

- `task`
  - 返回格式：`Precondition: ... Goal: ...`
- `to_dict()`
  - 转成 JSON 可序列化字典
- `to_prompt_text()`
  - 转成两行文本，便于调试或后续 prompt 拼接
- `memory_key_text()`
  - 生成可作为 memory key 的文本形式

### `PlannerResult`

字段：

- `is_goal_complete: bool`
  - 是否判定总目标已完成
- `completion_message: str`
  - 当 `complete_goal` 时的完成消息
- `subtasks: List[PlannerSubtask]`
  - 当返回 `set_tasks` 时的子任务列表
- `raw_response: str`
  - 模型原始文本响应
- `parse_error: Optional[str]`
  - 解析失败时的错误信息

两种典型结果：

1. 总目标已完成

```python
PlannerResult(
    is_goal_complete=True,
    completion_message="Contact has been created.",
    subtasks=[],
    raw_response='{"tool":"complete_goal","message":"Contact has been created."}',
    parse_error=None,
)
```

2. 生成下一轮任务

```python
PlannerResult(
    is_goal_complete=False,
    completion_message="",
    subtasks=[...],
    raw_response='{"tool":"set_tasks","tasks":[...]}',
    parse_error=None,
)
```

### `PlannerConfig`

当前字段：

- `max_subtasks`
  - planner 最多输出多少个子任务
- `max_ui_elements`
  - 当前配置项保留在 planner 中，但 observation 的截断主要由 adapter 侧控制
- `max_history_items`
  - prompt 中最多保留多少条 task history
- `max_memory_context_chars`
  - memory context 最大字符数
- `max_ui_json_chars`
  - UI JSON 最大字符数
- `temperature`
  - planner 调模型时使用的温度
- `default_actor_name`
  - planner 解析结果里默认填入的 agent 名称

真正会显著影响 prompt 长度或 planner 行为的字段主要是：

- `max_subtasks`
- `max_history_items`
- `max_memory_context_chars`
- `max_ui_json_chars`
- `temperature`

### `AndroidTaskPlanner`

#### `plan(...)`

签名：

```python
plan(
    user_goal: str,
    observation: dict,
    task_history: list[dict] | None = None,
    memory_context: str = "",
) -> PlannerResult
```

作用：

- 高层入口
- 内部会先调用 `build_messages(...)`
- 再调用 client 的 `generate(...)`
- 最后调用 `parse_response(...)`

#### `build_messages(...)`

作用：

- 构造 multimodal chat messages
- 当前固定返回两条 message：
  - `system`
  - `user`

其中 `user.content` 是一个 list，包含：

- 一段文本 prompt
- 一张图片，优先 `labeled_screenshot_b64`

#### `messages_to_jsonable(...)`

作用：

- 将 messages 转成稳定可保存的 JSON 结构
- 供 `planner_smoke.py` 写入 `planner_messages.json`

#### `extract_user_text_prompt(...)`

作用：

- 从 multimodal messages 中提取 user 的纯文本 prompt
- 供 `planner_smoke.py` 写入 `planner_prompt.txt`

#### `parse_response(...)`

作用：

- 解析模型原始文本返回
- 支持：
  - `{"tool":"complete_goal","message":"..."}`
  - `{"tool":"set_tasks","tasks":[...]}`
- 兼容旧格式：
  - `set_tasks_with_agents`
  - `task_assignments`

#### `_build_system_prompt()` / `_build_user_prompt()`

这两个方法不是对外主入口，但调试时很重要：

- `_build_system_prompt()`
  - 生成 planner 的角色说明、输出约束
- `_build_user_prompt()`
  - 生成本轮 observation、history、memory context 对应的 user text

### `extract_json_object(...)`

作用：

- 从模型返回的自由文本中提取 planner JSON

为什么需要它：

- 模型不总是严格返回干净 JSON
- 真实调用中常见问题包括：
  - 返回尾部多一个 `"`
  - 返回 JSON string 包裹的 JSON object
  - 返回带嵌套数组/对象的 JSON

当前它兼容的“脏返回”包括：

- 标准 JSON object
- 结尾多一个孤立引号的 JSON
- JSON-encoded string
- 含嵌套对象的 JSON，使用平衡大括号方式提取

### `parse_precondition_goal(...)`

作用：

- 解析 `Precondition: ... Goal: ...` 形式的子任务字符串

预期输入：

```text
Precondition: None. Goal: Open the Phone app.
```

失败条件：

- 不含 `Precondition:`
- 不含 `Goal:`
- 两部分任一为空

## Input / Output Contract

### Planner 输入

`plan(...)` 的输入结构如下：

- `user_goal: str`
- `observation: dict`
- `task_history: list[dict]`
- `memory_context: str`

### Observation 至少应包含的字段

- `current_activity`
- `app_name`
- `screen_size`
- `ui_elements`
- `ui_description`
- `screenshot_b64`
- `labeled_screenshot_b64`

最小 observation 示例：

```python
observation = {
    "goal": "Create a new contact for Henry Ali. Their number is +10024525408.",
    "current_activity": "com.google.android.apps.nexuslauncher/com.google.android.apps.nexuslauncher.NexusLauncherActivity",
    "app_name": "com.google.android.apps.nexuslauncher",
    "screen_size": {"width": 1080, "height": 2400},
    "ui_elements": [
        {
            "index": 0,
            "text": "Phone",
            "content_description": None,
            "resource_name": None,
            "class_name": "android.widget.TextView",
            "bbox": [100, 2000, 220, 2140],
            "is_clickable": True,
            "is_editable": False,
            "is_enabled": True,
            "is_scrollable": False,
            "is_visible": True,
            "package_name": "com.google.android.apps.nexuslauncher",
            "raw": {}
        }
    ],
    "ui_description": "UI element 0: text='Phone', class_name='android.widget.TextView', is_clickable=True, is_editable=False, is_enabled=True, is_scrollable=False, bbox=(100, 2000, 220, 2140)",
    "valid_ui_indices": [0],
    "screenshot_b64": "...",
    "labeled_screenshot_b64": "...",
    "extra_state": {"step_id": 0, "orientation": 0},
}
```

### Planner 输出

当前标准输出只有两类：

1. `complete_goal`

```json
{"tool":"complete_goal","message":"Contact has been created."}
```

2. `set_tasks`

```json
{
  "tool": "set_tasks",
  "tasks": [
    {
      "task": "Precondition: None. Goal: Open the Phone app.",
      "reason": "The user needs to open the Phone app to add a new contact."
    }
  ]
}
```

最小 raw planner response 示例：

```text
{"tool":"set_tasks","tasks":[{"task":"Precondition: None. Goal: Open the Phone app.","reason":"The user needs to open the Phone app to add a new contact."}]}
```

`PlannerResult.to_dict()` 示例：

```python
{
    "is_goal_complete": False,
    "completion_message": "",
    "subtasks": [
        {
            "precondition": "None.",
            "goal": "Open the Phone app.",
            "reason": "The user needs to open the Phone app to add a new contact.",
            "agent": "android_actor",
        }
    ],
    "raw_response": "{\"tool\":\"set_tasks\",\"tasks\":[...]}",
    "parse_error": None,
}
```

## How To Run

### 1. 纯代码调用

最小 Python 示例：

```python
from dms_reproduction.agents.planner import AndroidTaskPlanner
from dms_reproduction.llm.base_client import OpenAICompatibleConfig
from dms_reproduction.llm.openai_compatible import OpenAICompatibleClient

client = OpenAICompatibleClient(
    OpenAICompatibleConfig(
        base_url="http://127.0.0.1:8000/v1",
        api_key="dms-qwen-secret",
        model="qwen2.5-vl-7b",
        max_tokens=512,
        timeout=120,
    )
)

planner = AndroidTaskPlanner(client)

observation = {
    "current_activity": "com.google.android.apps.nexuslauncher/com.google.android.apps.nexuslauncher.NexusLauncherActivity",
    "app_name": "com.google.android.apps.nexuslauncher",
    "screen_size": {"width": 1080, "height": 2400},
    "ui_elements": [],
    "ui_description": "No visible UI elements available.",
    "screenshot_b64": None,
    "labeled_screenshot_b64": None,
}

result = planner.plan(
    user_goal="Create a new contact for Henry Ali. Their number is +10024525408.",
    observation=observation,
    task_history=[],
    memory_context="",
)

print(result.to_dict())
```

### 2. 使用 smoke 脚本

常用命令：

```powershell
python scripts/planner_smoke.py --skip_emulator_launch
```

如果需要密码 tunnel：

```powershell
python scripts/planner_smoke.py --ssh_password 123456 --skip_emulator_launch
```

常用参数：

- `--task`
  - 指定 AndroidWorld 任务名，默认 `ContactsAddContact`
- `--skip_ssh_tunnel`
  - 跳过脚本内 tunnel 启动逻辑，适合已经手工建好 tunnel 的场景
- `--skip_emulator_launch`
  - 跳过脚本内模拟器启动逻辑，适合模拟器已经手工启动的场景
- `--ssh_password`
  - 使用 `paramiko` 在脚本内建立带密码的 SSH tunnel

### 3. 输出目录说明

每次运行会生成：

```text
planner_smoke_runs/<task_timestamp>/
```

其中关键文件用途如下：

- `observation.json`
  - 本次 planner 调用的 observation
- `planner_messages.json`
  - 发送给模型的完整 multimodal messages
- `planner_prompt.txt`
  - user 的纯文本 prompt
- `planner_raw_response.txt`
  - 模型原始返回
- `planner_result.json`
  - parser 解析后的结构化结果
- `meta.json`
  - 记录模型地址、tunnel 模式、错误信息、耗时等运行元数据

## Testing & Debugging

### 单测命令

```powershell
python -m unittest tests.test_planner
python -m unittest tests.test_planner_smoke_config
python -m unittest tests.test_observation_pipeline
```

### 每组测试在测什么

- `tests/test_planner.py`
  - planner prompt 构造
  - 单图消息约束
  - JSON 解析
  - 脏返回兼容
- `tests/test_planner_smoke_config.py`
  - smoke 脚本 CLI 参数
  - SSH tunnel 逻辑
  - 模拟器启动逻辑
  - meta 信息构造
- `tests/test_observation_pipeline.py`
  - observation adapter
  - UI element 过滤
  - schema 正确性

### 常见问题

#### 1. vLLM 限制 1 张图

如果 vLLM 配置了：

```text
--limit-mm-per-prompt image=1
```

那 planner 只能传 1 张图。当前实现默认只传：

- `labeled_screenshot_b64`

如果没有标注图，才退回传：

- `screenshot_b64`

#### 2. prompt 超过 context length

如果出现：

```text
Input length exceeds model's maximum context length
```

说明 prompt 太长，常见原因是：

- `ui_elements JSON` 太大
- `task_history` 太长
- `memory_context` 太长

排查方式：

- 打开 `planner_prompt.txt`
- 打开 `planner_messages.json`
- 先看 `Visible UI elements JSON` 是否过长

#### 3. raw response 尾部多引号导致 parse error

真实模型返回中可能出现：

```text
{"tool":"set_tasks","tasks":[...]}"
```

当前 `extract_json_object(...)` 已兼容：

- 尾部孤立引号
- JSON-encoded string
- 嵌套对象/数组

如果仍然 parse error，请优先检查：

- `planner_raw_response.txt`

#### 4. `--skip_emulator_launch` 的含义

`--skip_emulator_launch` 只适用于：

- 模拟器已经手工启动
- `adb devices` 里已经能看到 `emulator-5554`
- `grpc 8554` 端口已经 ready

否则脚本会在 AndroidWorld 接入前失败。

#### 5. snapshot warning 与 a11y retry warning

`Skipping app snapshot loading : Snapshot not found ...`
- 表示 AndroidWorld 没找到对应 app 的 snapshot
- 对单次 planner smoke 一般不是致命错误
- 对正式 benchmark 评测应补齐 snapshot，避免任务初始状态漂移

`Could not get a11y tree, retrying.`
- 表示本次 accessibility tree 获取失败过一次
- AndroidWorld 会自动重试
- 如果后续 observation 正常，一般可以先忽略
- 如果频繁出现并导致空 UI，则需要排查 a11y forwarding 稳定性

## 相关文件

- [dms_reproduction/agents/planner.py](/f:/baoyantest/dms/dms_reproduction/agents/planner.py)
- [dms_reproduction/llm/openai_compatible.py](/f:/baoyantest/dms/dms_reproduction/llm/openai_compatible.py)
- [dms_reproduction/envs/android_world_adapter.py](/f:/baoyantest/dms/dms_reproduction/envs/android_world_adapter.py)
- [scripts/planner_smoke.py](/f:/baoyantest/dms/scripts/planner_smoke.py)
- [tests/test_planner.py](/f:/baoyantest/dms/tests/test_planner.py)
- [tests/test_planner_smoke_config.py](/f:/baoyantest/dms/tests/test_planner_smoke_config.py)
- [tests/test_observation_pipeline.py](/f:/baoyantest/dms/tests/test_observation_pipeline.py)
