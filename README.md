This repository is built on top of AndroidWorld. The original AndroidWorld code is used as the GUI interaction environment, while the main reproduction code is implemented in the following files and directories.

## Core Files

- `A_zero-shot_vlm_baseline.py`: zero-shot VLM baseline
- `dms/`: memory construction, retrieval, survival value, and pruning modules
- `docs/architecture_overview.md`: project architecture overview
- `docs/planner_module.md`: planner module notes
- `DMS.pdf`: reference paper
- `ufo/`: related agent implementation references

## Current Agent Status

For the current Android GUI agent status, handoff notes, and latest blocking issues, start with:

1. [docs/android_gui_agent_handoff.md](docs/android_gui_agent_handoff.md)
2. [docs/current_status.md](docs/current_status.md)
3. [docs/architecture_overview.md](docs/architecture_overview.md)
4. [docs/planner_module.md](docs/planner_module.md)
5. [scripts/task_loop_smoke.py](scripts/task_loop_smoke.py)

Current main loop:

- `planner -> android_actor -> task_runner -> task_loop_smoke.py`
- mock-based tests and smoke artifacts are already in place
- memory is not yet wired into the runtime loop

Latest smoke sample:

- [task_loop_smoke_runs/ContactsAddContact_20260515_220450/run_summary.md](task_loop_smoke_runs/ContactsAddContact_20260515_220450/run_summary.md)

Current top issues:

1. `success override` can still end with a degraded `system-ui-only` observation.
2. planner can still regress to low-level atomic goals near the contact-creation entry point.
3. planner grounding veto prevents bad actions, but recovery after veto is still weak.

