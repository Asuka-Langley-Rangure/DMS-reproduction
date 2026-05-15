## 1.虚拟机启动
命令行启动项目目录下

```
start_androidworld_emulator.bat
```

### 2.验证设备连接

```
(android_world) F:\baoyantest\dms\android_world>"D:\Android\Sdk\platform-tools\adb.exe" devices            
List of devices attached
emulator-5554   device
```

### 3.验证Python能否和emulator链接

### 4.服务器模型部署

#### 模型所在文件夹

```
(qwen25vl) chencen@test-G7466-M6:/data1/chencen/dms_qwen$  
```

#### 模型对应的虚拟环境

```shell
# conda activate /data1/chencen/dms_qwen/envs/qwen25vl 
source /data1/chencen/dms_qwen/scripts/activate_qwen25vl.sh
```

#### vLLM

> vLLM 可以理解成一个**大模型推理服务器 / 推理加速框架**。
> 
> 它不是模型，也不是 Qwen 的一部分，而是负责把 **Qwen2.5-VL-7B-Instruct 加载到 GPU 上，然后对外提供 API 调用服务**。
> 
> 你现在的架构里，vLLM 的位置是：
> 
> ```text
> Windows 本机
> AndroidWorld / Zero-shot / Static Memory / DMS
>         │
>         │ HTTP 请求
>         ▼
> 实验室服务器
> vLLM 服务
>         │
>         ▼
> Qwen2.5-VL-7B-Instruct 权重
>         │
>         ▼
> RTX 4090 GPU 推理
> ```
> 
> ## 它具体做什么？
> 
> ### 1. 加载模型到 GPU
> 
> 比如我们启动：
> 
> ```bash
> vllm serve /data1/chencen/dms_qwen/models/Qwen2.5-VL-7B-Instruct
> ```
> 
> vLLM 会把 Qwen2.5-VL-7B 模型加载到服务器的 RTX 4090 显存里。
> 
> 之后模型会一直驻留在 GPU 中，不需要每次请求都重新加载。
> 
> ---
> 
> ### 2. 提供 HTTP API
> 
> 启动成功后，它会开放类似这样的接口：
> 
> ```text
> http://127.0.0.1:8000/v1/chat/completions
> ```
> 
> 你的 Windows 本机就可以像调用 OpenAI API 一样调用它：
> 
> ```python
> from openai import OpenAI
> 
> client = OpenAI(
>     base_url="http://127.0.0.1:8000/v1",
>     api_key="dms-qwen-secret"
> )
> ```
> 
> 也就是说，vLLM 会把本地部署的 Qwen 模型包装成一个“类 OpenAI 接口”。
> 
> ---
> 
> ### 3. 加速推理
> 
> 如果你直接用 Transformers 调用模型，大致是：
> 
> ```python
> model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
> ```
> 
> 这种方式适合单次测试，但缺点是：
> 
> ```text
> 每个脚本都要加载模型
> 不方便远程调用
> 多次请求管理麻烦
> 并发能力弱
> 长时间实验不太稳定
> ```
> 
> vLLM 则更适合服务化部署，它会自动管理：
> 
> ```text
> GPU 显存
> KV cache
> 请求队列
> batching
> API 服务
> 多轮推理
> ```
> 
> 你后面跑 AndroidWorld 时，每一个任务、每一步操作都可能要调用一次 VLM。用 vLLM 会比“每次 Python 脚本直接加载模型”更合适。
> 
> ---
> 
> ## 为什么我们这里推荐 vLLM？
> 
> 因为你的目标不是简单跑一次图片问答，而是要做：
> 
> ```text
> AndroidWorld 多任务测试
> Zero-shot baseline
> 静态记忆 baseline
> DMS 记忆系统
> 多轮 trial 对比
> 大量截图 + prompt 请求
> ```
> 
> 这些实验会产生很多模型调用。
> 
> 所以更合理的方式是：
> 
> ```text
> 服务器长期启动一个 Qwen2.5-VL API 服务
> Windows 本机每次需要模型时就发 HTTP 请求
> ```
> 
> 这样你的 DMS、Memory Bank、Pruning、日志记录都可以放在 Windows 上，而服务器只负责模型推理。
> 
> ---
> 
> ## 可以不用 vLLM 吗？
> 
> 可以，但不推荐。
> 
> 你有三种方案：
> 
> | 方案                | 优点                | 缺点                |
> | ----------------- | ----------------- | ----------------- |
> | Transformers 直接推理 | 最容易理解，适合单次测试      | 不适合远程 API，不适合大量实验 |
> | vLLM              | 适合部署 API，速度快，调用方便 | 安装稍微复杂            |
> | LMDeploy / SGLang | 也适合部署             | 需要额外适配和学习         |
> 
> 对于你这次考核，**vLLM 是最适合的折中方案**。
> 
> ---
> 
> 一句话总结：
> 
> **vLLM 就是把服务器上的 Qwen2.5-VL-7B 变成一个可以被 Windows 远程调用的高性能 API 服务。**

#### 模型下载后创建对应的软连接

```shell
(qwen25vl) chencen@test-G7466-M6:/data1/chencen/dms_qwen$ MODEL_SRC=/data1/chencen/dms_qwen/models/Qwen/Qwen2___5-VL-7B-Instruct
(qwen25vl) chencen@test-G7466-M6:/data1/chencen/dms_qwen$ MODEL_DST=/data1/chencen/dms_qwen/models/Qwen2.5-VL-7B-Instruct
(qwen25vl) chencen@test-G7466-M6:/data1/chencen/dms_qwen$ rm -f "$MODEL_DST"
(qwen25vl) chencen@test-G7466-M6:/data1/chencen/dms_qwen$ ln -s "$MODEL_SRC" "$MODEL_DST"
(qwen25vl) chencen@test-G7466-M6:/data1/chencen/dms_qwen$ ls -ld "$MODEL_DST"
lrwxrwxrwx 1 chencen chencen 60  5月 10 17:33 /data1/chencen/dms_qwen/models/Qwen2.5-VL-7B-Instruct -> /data1/chencen/dms_qwen/models/Qwen/Qwen2___5-VL-7B-Instruct
(qwen25vl) chencen@test-G7466-M6:/data1/chencen/dms_qwen$ readlink -f "$MODEL_DST"
/data1/chencen/dms_qwen/models/Qwen/Qwen2___5-VL-7B-Instruct
```

#### vLLM服务器参数说明

[vLLM server arguments | Red Hat AI Inference Server | 3.1 | Red Hat Documentation](https://docs.redhat.com/de/documentation/red_hat_ai_inference_server/3.1/html-single/vllm_server_arguments/index?utm_source=chatgpt.com)

> 可以。我们这条命令的本质是：**先进入 Qwen2.5-VL 的部署环境，然后指定 GPU 7，把本地模型权重用 vLLM 启动成一个 OpenAI-compatible API 服务**。vLLM 官方说明它提供 HTTP server，支持 OpenAI 风格的 Chat API，并可用 `vllm serve` 启动服务。([vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html "OpenAI-Compatible Server - vLLM"))
> 
> 原命令是：
> 
> ```bash
> source /data1/chencen/dms_qwen/scripts/activate_qwen25vl.sh
> 
> CUDA_VISIBLE_DEVICES=7 \
> vllm serve /data1/chencen/dms_qwen/models/Qwen2.5-VL-7B-Instruct \
>   --host 127.0.0.1 \
>   --port 8007 \
>   --served-model-name qwen2.5-vl-7b \
>   --dtype bfloat16 \
>   --max-model-len 8192 \
>   --limit-mm-per-prompt image=1,video=0 \
>   --gpu-memory-utilization 0.85 \
>   --api-key dms-qwen-secret \
>   --trust-remote-code
> ```
> 
> ## 逐项解释
> 
> ### `source /data1/chencen/dms_qwen/scripts/activate_qwen25vl.sh`
> 
> 这一步是加载我们之前写好的环境脚本。它主要做几件事：
> 
> ```bash
> conda activate /data1/chencen/dms_qwen/envs/qwen25vl
> ```
> 
> 激活你刚刚创建的 `qwen25vl` 环境。
> 
> ```bash
> export HF_HOME=/data1/chencen/dms_qwen/hf_cache
> export PIP_CACHE_DIR=/data1/chencen/dms_qwen/pip_cache
> ```
> 
> 把 Hugging Face 缓存、pip 缓存等都指向 `/data1`，避免写到已经快满的根分区。
> 
> 这一步可以理解为：
> 
> > 进入正确的 Python 环境，并设置好模型部署相关路径。
> 
> ---
> 
> ### `CUDA_VISIBLE_DEVICES=7 \`
> 
> 这表示：**当前这条命令只让程序看到物理 GPU 7**。
> 
> 你的服务器有 8 张卡，编号是 0 到 7。我们选择 GPU 7，是因为它当前基本空闲。
> 
> 需要注意：
> 
> ```text
> 物理 GPU 7  →  程序内部看到的 cuda:0
> ```
> 
> 也就是说，vLLM 进程里如果打印 `cuda:0`，不代表它用了物理 GPU 0，而是因为你通过 `CUDA_VISIBLE_DEVICES=7` 只暴露了一张卡给它。
> 
> 后面的反斜杠 `\` 表示命令还没结束，下一行继续。
> 
> ---
> 
> ### `vllm serve /data1/chencen/dms_qwen/models/Qwen2.5-VL-7B-Instruct`
> 
> 这是真正启动模型服务的主体。
> 
> ```bash
> vllm serve
> ```
> 
> 表示启动 vLLM 的 OpenAI-compatible server。vLLM 文档给出的启动方式也是 `vllm serve <model>`，然后用 OpenAI Python client 访问。([vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html "OpenAI-Compatible Server - vLLM"))
> 
> ```bash
> /data1/chencen/dms_qwen/models/Qwen2.5-VL-7B-Instruct
> ```
> 
> 是模型路径。这里指向我们刚刚下载好的 Qwen2.5-VL-7B-Instruct 权重目录。
> 
> 这一步可以理解为：
> 
> > 把这个目录里的 Qwen2.5-VL-7B 模型加载进 GPU，并启动一个 HTTP 推理服务。
> 
> ---
> 
> ### `--host 127.0.0.1`
> 
> 表示服务只监听服务器本机地址。
> 
> 也就是说，服务器自己可以访问：
> 
> ```text
> http://127.0.0.1:8007
> ```
> 
> 但外部机器不能直接访问这个端口。
> 
> 这正好符合我们的方案：**不直接暴露端口，而是用 Windows 通过 SSH 隧道转发访问**。这样更安全。
> 
> 如果写成：
> 
> ```bash
> --host 0.0.0.0
> ```
> 
> 就表示允许其他机器直接访问服务器的这个端口，但这需要防火墙、实验室网络配置支持，而且安全风险更高。
> 
> ---
> 
> ### `--port 8007`
> 
> 表示服务端口是 `8007`。
> 
> 我们选择 `8007` 是为了和 GPU 7 对应，方便记忆：
> 
> ```text
> GPU 7  →  port 8007
> ```
> 
> 后面在服务器本机测试时会访问：
> 
> ```bash
> curl http://127.0.0.1:8007/v1/models
> ```
> 
> Windows 端用 SSH 隧道后，可以把本地 `8000` 转发到服务器 `8007`：
> 
> ```powershell
> ssh -N -L 8000:127.0.0.1:8007 chencen@test-G7466-M6
> ```
> 
> 这样 Windows 访问：
> 
> ```text
> http://127.0.0.1:8000/v1
> ```
> 
> 实际就是访问服务器上的：
> 
> ```text
> http://127.0.0.1:8007/v1
> ```
> 
> ---
> 
> ### `--served-model-name qwen2.5-vl-7b`
> 
> 这是对外暴露的模型名字。
> 
> 也就是说，后面你的 Python 客户端里要这样写：
> 
> ```python
> response = client.chat.completions.create(
>     model="qwen2.5-vl-7b",
>     ...
> )
> ```
> 
> 这个名字不一定要和模型目录名完全一样，只要你启动服务和客户端调用时一致即可。
> 
> 如果你这里写成：
> 
> ```bash
> --served-model-name dms-qwen
> ```
> 
> 那客户端里就要写：
> 
> ```python
> model="dms-qwen"
> ```
> 
> ---
> 
> ### `--dtype bfloat16`
> 
> 表示模型推理时使用 `bfloat16` 精度。
> 
> 它的作用是降低显存占用、提高推理效率，同时相较 `float32` 损失很小。4090 支持 BF16，所以这里可以用。
> 
> 常见选择有：
> 
> ```text
> float32   显存大，速度慢，一般不用
> float16   常见半精度
> bfloat16  更稳一些，4090 支持
> auto      让 vLLM 自动判断
> ```
> 
> 如果后面遇到兼容问题，可以改成：
> 
> ```bash
> --dtype auto
> ```
> 
> ---
> 
> ### `--max-model-len 8192`
> 
> 表示限制模型服务允许的最大上下文长度为 **8192 tokens**。
> 
> 这会影响两件事：
> 
> ```text
> 越大：能塞更长的 prompt、UI 元素、历史记忆，但显存占用更高
> 越小：显存更省，但长任务上下文容易不够
> ```
> 
> vLLM 文档也说明，`--max-model-len` 是模型最大上下文长度限制，设置它可以避免模型默认上下文过长带来的显存问题。([红帽文档](https://docs.redhat.com/de/documentation/red_hat_ai_inference_server/3.1/html-single/vllm_server_arguments/index?utm_source=chatgpt.com "vLLM server arguments | Red Hat AI Inference Server | 3.1"))
> 
> 对你当前阶段，8192 是比较合理的起步值。因为 AndroidWorld 每步会传：
> 
> ```text
> 任务目标
> 当前截图
> UI elements
> 历史动作
> 记忆检索结果
> ```
> 
> 如果后面 DMS prompt 变长，可以再考虑调到：
> 
> ```bash
> --max-model-len 16384
> ```
> 
> 但 7B + 4090 单卡下，8192 更稳。
> 
> ---
> 
> ### `--limit-mm-per-prompt image=1,video=0`
> 
> 这个参数是限制每个请求里最多能传多少多模态输入。
> 
> vLLM 文档里说，`--limit-mm-per-prompt` 用于限制每个 prompt 中每种模态允许的输入数量，支持用 JSON 或键值形式传入。([vLLM](https://docs.vllm.ai/en/stable/cli/serve/ "vllm serve - vLLM"))
> 
> 这里：
> 
> ```text
> image=1
> ```
> 
> 表示每次请求最多传 1 张图片。
> 
> ```text
> video=0
> ```
> 
> 表示不允许传视频。
> 
> 这很符合我们的 AndroidWorld 场景，因为每一步通常只需要传当前屏幕截图：
> 
> ```text
> 当前 screenshot + UI elements 文本
> ```
> 
> 不需要传视频。
> 
> 如果以后你想让模型同时看“前一帧截图 + 当前截图”，可以改成：
> 
> ```bash
> --limit-mm-per-prompt image=2,video=0
> ```
> 
> 但当前先用 `image=1` 更省显存、更稳定。
> 
> 有些版本的 vLLM 对这个参数格式更严格，如果报错，可以换成 JSON 形式：
> 
> ```bash
> --limit-mm-per-prompt '{"image": 1, "video": 0}'
> ```
> 
> ---
> 
> ### `--gpu-memory-utilization 0.85`
> 
> 表示 vLLM 最多使用这张 GPU 显存的 85%。
> 
> vLLM 文档中这个参数的含义是：限制当前 vLLM 实例可使用的 GPU 显存比例，取值范围是 0 到 1；比如 0.5 表示使用约 50% 显存。([vLLM](https://docs.vllm.ai/en/stable/cli/serve/ "vllm serve - vLLM"))
> 
> 你的 GPU 7 是 RTX 4090，显存约 49GB，所以：
> 
> ```text
> 49GB × 0.85 ≈ 41.6GB
> ```
> 
> vLLM 会在这个范围内分配模型权重、KV cache、多模态处理缓存等。
> 
> 如果太高，比如 `0.95`，可能启动时看起来能跑，但之后请求稍微大一点就 OOM。  
> 如果太低，比如 `0.5`，可能 KV cache 太小，吞吐下降，甚至长 prompt 不够。
> 
> 所以 `0.85` 是一个比较稳的起步值。
> 
> ---
> 
> ### `--api-key dms-qwen-secret`
> 
> 给这个 API 服务设置一个简单访问密钥。
> 
> 后面客户端调用时要写：
> 
> ```python
> client = OpenAI(
>     base_url="http://127.0.0.1:8000/v1",
>     api_key="dms-qwen-secret",
> )
> ```
> 
> vLLM 官方示例里也用 `--api-key token-abc123` 启动服务，并在 OpenAI Python client 里填同一个 `api_key`。([vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html "OpenAI-Compatible Server - vLLM"))
> 
> 因为我们服务只监听 `127.0.0.1`，再配合 SSH 隧道，这个密钥主要是防误调用，不是完整的安全体系。
> 
> ---
> 
> ### `--trust-remote-code`
> 
> 允许加载模型仓库里自定义的 Python 代码。
> 
> 有些模型，尤其是多模态模型，可能包含特殊的模型结构、processor、视觉编码逻辑，Transformers/vLLM 需要信任并执行这些代码才能正确加载。
> 
> 这个参数的风险是：如果模型来源不可信，远程代码可能有安全问题。  
> 但我们这里加载的是 Qwen 官方模型，并且模型已经下载到本地路径，所以可以接受。
> 
> ---
> 
> ## 总结成一句话
> 
> 这条命令等价于：
> 
> ```text
> 进入 qwen25vl 环境，
> 只使用物理 GPU 7，
> 把本地 Qwen2.5-VL-7B-Instruct 模型加载到 GPU，
> 在服务器本机 127.0.0.1:8007 启动一个 OpenAI 风格的 HTTP API，
> 对外模型名叫 qwen2.5-vl-7b，
> 每次请求最多 1 张图片，
> 上下文长度限制为 8192 tokens，
> 最多使用 85% GPU 显存，
> 并要求客户端用 dms-qwen-secret 作为 API key。
> ```
> 
> 你现在可以先用这条命令**前台启动一次**，这样如果参数不兼容，我们能直接看到报错并修改。

### 5.本地通过ssh隧道同服务器大模型联通

#### 使用powershell建立ssh隧道

```powershell
ssh -N -L 8000:127.0.0.1:8007 chencen@114.212.165.149
```

#### 测试API

```powershell
curl.exe http://127.0.0.1:8000/v1/models -H "Authorization: Bearer dms-qwen-secret"
```


