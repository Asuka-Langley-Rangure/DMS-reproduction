This repository is built on top of AndroidWorld. The original AndroidWorld code is used as the GUI interaction environment, while the main reproduction code is implemented in the following files:

- `A_zero-shot_vlm_baseline.py`: zero-shot VLM baseline
- `B_static_memory_baseline.py`: static memory baseline
- `C_dms_agent.py`: DMS-based agent
- `android_world/agents/my_dms_agent.py`: customized DMS agent implementation
- `dms/`: memory construction, retrieval, survival value, and pruning modules