# 模型调研记录

调研日期：2026-05-20

本文只服务第一轮 `P14 / P14-fast / P8 / FMT` baseline 选型。约束来自 `plan.md`、`docs/plans/agent-task-list.md`、`docs/contract/prompt-baseline.md`、`docs/contract/eval-plan.md`：

- 最终推理系统参与模型总参数量必须小于 20B。
- 第一轮不包含 Qwen3.5-9B；若提及，仅作为暂缓/备选。
- 优先验证 license、参数量、vLLM / Transformers 支持、thinking / non-thinking 模式、量化版本。
- Prompt baseline 目标是稳定输出最终 JSON；训练路线后续在 8B/9B 级模型中选 reasoner 基座。

## 1. 当前可用模型核验

| 模型 | License | 参数量 | Transformers 支持 | vLLM 支持 | thinking / non-thinking | 量化版本 | 结论 |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `Qwen/Qwen3-14B` | Apache-2.0 | 14.8B | HF card 给出 `AutoModelForCausalLM` 用法；Qwen 文档要求较新 Transformers | HF card 和 Qwen 文档给出 `vllm serve Qwen/Qwen3-14B` | 支持硬切换 `enable_thinking=True/False`，也支持 `/think`、`/no_think` 软切换 | 官方 `Qwen/Qwen3-14B-AWQ`；Qwen 文档说明 Qwen3 有 FP8 和 AWQ 预量化 | 第一轮 P14 主候选 |
| `Qwen/Qwen3-14B-AWQ` | Apache-2.0 | 14.8B，AWQ 4-bit | HF card 给出 Transformers 用法 | Qwen 文档给出 AWQ 的 vLLM serve 方式 | 与 Qwen3-14B 相同，建议 first-round 使用 non-thinking | 官方 AWQ 4-bit；同系列还有 FP8 | 第一轮 P14-fast 主候选 |
| `Qwen/Qwen3-8B` | Apache-2.0 | 8.2B | HF card 给出 Transformers 用法；要求较新 Transformers | HF card 和 vLLM supported models 均支持 Qwen3 | 支持 `enable_thinking=True/False`；non-thinking 不输出 `<think>` | 官方 `Qwen/Qwen3-8B-AWQ`；Qwen 文档说明 FP8/AWQ | 第一轮 P8 主候选，后续 BC8 基座优先 |
| `Qwen/Qwen3-8B-AWQ` | Apache-2.0 | 8.2B，AWQ 4-bit | HF card 给出 Transformers 用法 | Qwen 文档给出 AWQ 的 vLLM serve 方式 | 与 Qwen3-8B 相同 | 官方 AWQ 4-bit | 第一轮 P8-fast 可选，用于速度/质量对照 |
| `internlm/internlm3-8b-instruct` | Apache-2.0 | 8B | HF card 提供 Transformers / 自定义代码用法 | vLLM supported models 列出 `InternLM3ForCausalLM` 和该模型 | 支持 deep thinking mode 与 normal response mode | 官方未见 AWQ/GPTQ；HF 有 GGUF 等社区量化，需单独验收 | 第一轮 P8 对照候选 |
| `google/gemma-4-E4B-it` / `google/gemma-4-E4B` | Apache-2.0 | 4.5B effective，8B with embeddings | HF card 给出 `AutoModelForImageTextToText`；Google blog/HF card 说明开放权重 | vLLM supported models 列出 `Gemma4ForConditionalGeneration` | Gemma 4 card 写明 configurable thinking modes | 官方 card 未列 AWQ/GPTQ；HF 上有 GGUF / LiteRT 等社区或边缘量化，vLLM 量化需实测 | 第一轮 FMT 主候选 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | Qwen 派生权重原始 Apache-2.0；DeepSeek card 说明 R1 蒸馏 | 14B | HF card 给出 Transformers 用法 | HF card 给出 `vllm serve`；Qwen2 架构在 vLLM 支持范围 | reasoning-distill，默认长推理；没有 Qwen3 那种明确 non-thinking 硬开关 | 官方未见 AWQ/GPTQ；社区 GGUF/GPTQ/AWQ 需逐个验收 | 暂缓：JSON-only baseline 风险高 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 同上 | 7B | HF card 给出 Transformers 用法 | HF card 给出 `vllm serve` | reasoning-distill，默认长推理；无明确 non-thinking 硬开关 | 官方未见 AWQ/GPTQ；社区量化需验收 | 暂缓：可做推理型 P8 对照，但不进第一轮主表 |
| `microsoft/phi-4` | MIT | 14B | HF card 给出 Transformers 用法 | vLLM supported models 将 Phi-4 归入 `Phi3ForCausalLM` | 标准 instruct / chat，不是同一模型内 thinking 切换 | 官方未见 AWQ/GPTQ；社区 GGUF 常见，vLLM 量化需验收 | 暂缓：中文古诗词能力不如中文系模型可预期 |
| `microsoft/Phi-4-mini-instruct` | MIT | 3.8B 级 | HF card 给出 Transformers 用法 | HF card 给出 `vllm serve` | 标准 instruct；无同一模型 thinking 切换 | HF 有 Browse Quantizations；官方 AWQ/GPTQ 未作为主线确认 | 暂缓：可做轻量 formatter 备选，但中文与格式修复需先 smoke |
| `Qwen/Qwen3.5-9B` | Apache-2.0 | card 写 Language Model 9B；HF metadata 显示约 10B params | HF card 要求最新 Transformers / serving | HF card 和 vLLM supported models 支持 `Qwen3_5ForConditionalGeneration`；text-only 可用 `--language-model-only` | 默认 thinking；card 给出 thinking 与 instruct/non-thinking 采样建议 | HF 有大量社区 GGUF/AWQ/MLX 量化；官方量化需后续确认 | 按项目约束第一轮暂缓 |
| `openai/gpt-oss-20b` | Apache-2.0 | 20B | HF card 可用 pipeline；OpenAI 官方称开放权重 | vLLM supported models 支持 `GptOssForCausalLM`；vLLM 有专门 recipe | open-weight reasoning model；需 Harmony/推理格式适配 | 原生面向低显存部署；量化/精度路径不同于 AWQ/GPTQ | 暂缓：严格“小于 20B”下单模型已贴边，且中文古诗词风险未知 |

## 2. 分组候选清单

### P14：14B prompt baseline

第一轮只放：

| 实验 ID 建议 | 模型 | 后端 | 模式 | 理由 |
| --- | --- | --- | --- | --- |
| `P14-qwen3-14b-bf16-vllm-nothink-few5` | `Qwen/Qwen3-14B` | vLLM | non-thinking | 14.8B 小于 20B；Apache-2.0；中文、多语言和 JSON 指令跟随更贴近任务；non-thinking 有助于减少 `<think>` 与额外文本 |
| `P14-qwen3-14b-bf16-vllm-nothink-zero` | `Qwen/Qwen3-14B` | vLLM | non-thinking | 与 few-shot 对照，测 prompt 依赖 |

暂缓：

- `microsoft/phi-4`：MIT、14B、vLLM 支持，但中文古诗词、古汉语翻译和情感选项不是强项，作为非中文 14B 对照暂缓。
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`：推理强，但 reasoning 输出倾向与“只输出最终 JSON”冲突，先不放 P14 主结果。

### P14-fast：14B 量化 prompt baseline

第一轮只放：

| 实验 ID 建议 | 模型 | 后端 | 模式 | 理由 |
| --- | --- | --- | --- | --- |
| `P14-fast-qwen3-14b-awq4-vllm-nothink-few5` | `Qwen/Qwen3-14B-AWQ` | vLLM | non-thinking | 官方 AWQ 4-bit；同一 Qwen3-14B 语义能力轴，便于量化速度/质量对照 |
| `P14-fast-qwen3-14b-awq4-vllm-nothink-zero` | `Qwen/Qwen3-14B-AWQ` | vLLM | non-thinking | 测量量化下 zero-shot 格式稳定性 |

暂缓：

- Qwen3-14B FP8：官方有 FP8 路径，但 Qwen 文档提示 FP8 block-wise 依赖较新 NVIDIA GPU 架构；若服务器 GPU 不满足，AWQ 更稳。

### P8 sweep：8B 级 reasoner 基座横扫

第一轮放：

| 实验 ID 建议 | 模型 | 后端 | 模式 | 理由 |
| --- | --- | --- | --- | --- |
| `P8-qwen3-8b-bf16-vllm-nothink-few3` | `Qwen/Qwen3-8B` | vLLM | non-thinking | 主学生基座优先项；8.2B，Apache-2.0，官方支持 thinking/non-thinking，后续 QLoRA 路线清晰 |
| `P8-qwen3-8b-awq4-vllm-nothink-few3` | `Qwen/Qwen3-8B-AWQ` | vLLM | non-thinking | 速度/显存对照；若 bf16 延迟不可接受，可作为 prompt baseline 低成本版本 |
| `P8-internlm3-8b-bf16-vllm-normal-few3` | `internlm/internlm3-8b-instruct` | vLLM | normal response | Apache-2.0；中文模型；支持 normal/deep thinking，可验证非 Qwen 中文基座 |
| `P8-internlm3-8b-bf16-vllm-think-few3` | `internlm/internlm3-8b-instruct` | vLLM | deep thinking | 只做小样本 smoke，观察 structured evidence / 情感题是否受益，不覆盖主 nothink 结果 |

暂缓：

- `Qwen/Qwen3.5-9B`：当前已可用，HF card 显示 Apache-2.0、9B/约 10B、vLLM 与 Transformers 支持，默认 thinking 且支持 instruct/non-thinking 采样；但项目明确第一轮不包含 Qwen3.5-9B，因此只列为第二轮备选。
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`：适合推理题对照，但长 reasoning 容易破坏 final JSON-only baseline；先等 Qwen3 / InternLM3 结果。
- `microsoft/Phi-4-mini-instruct`：轻量、MIT、vLLM 支持，但中文古诗词弱项风险高；可在 FMT 不稳定时做 JSON 修复 smoke。

### FMT：formatter / verifier baseline

第一轮放：

| 实验 ID 建议 | 模型 | 后端 | 模式 | 理由 |
| --- | --- | --- | --- | --- |
| `FMT-gemma4-e4b-bf16-transformers-nothink-jsonfix` | `google/gemma-4-E4B-it` | Transformers | non-thinking / direct JSON | 4.5B effective，8B with embeddings；Apache-2.0；参数预算可与 8.2B reasoner 组合为约 16.2B，总量小于 20B |
| `FMT-qwen3-8b-bf16-vllm-nothink-jsonfix` | `Qwen/Qwen3-8B` | vLLM | non-thinking | 与主 reasoner 同族，JSON 指令和中文稳定性预期更强；与 Qwen3-8B reasoner 组合约 16.4B，小于 20B |
| `FMT-qwen3-8b-awq4-vllm-nothink-jsonfix` | `Qwen/Qwen3-8B-AWQ` | vLLM | non-thinking | 若 formatter 延迟或显存压力大，测 AWQ 质量损失 |

暂缓：

- `google/gemma-3n-E4B-it`：vLLM 支持 Gemma3n，但 vLLM 文档提示 V1、`timm` 依赖和 PLE/缓存优化不足；Gemma 4 E4B 当前更贴合计划中的 “Gemma 4 E4B”。
- `microsoft/Phi-4-mini-instruct`：可做低参 JSON fixer smoke，但第一轮 formatter 主线先比较 Gemma 4 E4B 与 Qwen3-8B。

## 3. 第一轮进入/暂缓列表

进入第一轮：

| 组别 | 模型 | 关键配置 |
| --- | --- | --- |
| P14 | `Qwen/Qwen3-14B` | bf16，vLLM，non-thinking，zero/few-shot |
| P14-fast | `Qwen/Qwen3-14B-AWQ` | AWQ 4-bit，vLLM，non-thinking，zero/few-shot |
| P8 | `Qwen/Qwen3-8B` | bf16，vLLM，non-thinking，few3/zero |
| P8 | `Qwen/Qwen3-8B-AWQ` | AWQ 4-bit，vLLM，non-thinking，few3 |
| P8 | `internlm/internlm3-8b-instruct` | bf16，vLLM，normal response；deep thinking 只 smoke |
| FMT | `google/gemma-4-E4B-it` | Transformers 优先，direct JSON / non-thinking |
| FMT | `Qwen/Qwen3-8B` | vLLM，non-thinking，jsonfix |
| FMT | `Qwen/Qwen3-8B-AWQ` | vLLM，AWQ 4-bit，non-thinking，jsonfix |

暂缓：

| 模型 | 暂缓原因 |
| --- | --- |
| `Qwen/Qwen3.5-9B` | 虽然当前可用且 license/后端支持良好，但项目明确第一轮不包含；第二轮可作为 9B reasoner 备选 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | reasoning 输出倾向强，可能增加 JSON 错误率；P14 先用 Qwen3-14B 建上限 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 同上；可在情感/推理题弱时作为 P8 补充对照 |
| `microsoft/phi-4` | 14B/MIT/vLLM 支持，但中文古诗词任务针对性不足 |
| `microsoft/Phi-4-mini-instruct` | 轻量但中文和格式修复收益未知；FMT 主线先测 Gemma 4 E4B/Qwen3-8B |
| `openai/gpt-oss-20b` | Apache-2.0、vLLM 支持，但 20B 与“小于 20B”约束贴边，无法与 formatter 组合；中文古诗词能力待验证 |

## 4. 关键结论与来源

1. Qwen3-14B / 8B 是第一轮主轴：HF model card 确认 Apache-2.0、参数量、Transformers/vLLM 用法；Qwen 文档确认 vLLM reasoning parser、non-thinking chat template、FP8/AWQ 预量化。来源：[Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)、[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)、[Qwen vLLM 文档](https://qwen.readthedocs.io/en/stable/deployment/vllm.html)、[Qwen Transformers 文档](https://qwen.readthedocs.io/en/stable/inference/transformers.html)。
2. Qwen3-14B-AWQ / Qwen3-8B-AWQ 可作为官方 AWQ 4-bit 对照：HF card 标注 AWQ 4-bit，Qwen 文档给出相同 vLLM serve 路径。来源：[Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ)、[Qwen3-8B-AWQ](https://huggingface.co/Qwen/Qwen3-8B-AWQ)。
3. InternLM3-8B-Instruct 适合作为 P8 中文对照：HF card 确认 8B、Apache-2.0、deep thinking / normal response；vLLM supported models 列出 `InternLM3ForCausalLM`。来源：[InternLM3-8B-Instruct](https://huggingface.co/internlm/internlm3-8b-instruct)、[vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/)。
4. Gemma 4 E4B 是 FMT 主候选：HF card 确认 Apache-2.0、E4B 为 4.5B effective / 8B with embeddings、可配置 thinking modes；Google blog 确认 E4B/E2B 权重发布；vLLM supported models 列出 `Gemma4ForConditionalGeneration`。来源：[Gemma 4 E4B card](https://huggingface.co/google/gemma-4-E4B)、[Google Gemma 4 blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)、[vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/)。
5. Qwen3.5-9B 只作为第二轮备选：HF card 确认 Apache-2.0、9B、默认 thinking、vLLM/Transformers 支持；vLLM supported models 已列 `Qwen3_5ForConditionalGeneration`。但项目约束要求第一轮不包含。来源：[Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B)、[vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/)。
6. DeepSeek-R1-Distill-Qwen 系列暂缓：HF card 确认 Qwen 派生权重和 vLLM 用法，但其 reasoning-distill 行为更可能产生长推理文本，需等 JSON-only baseline 稳定后再测。来源：[DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)、[DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)。
7. Phi-4 / Phi-4-mini 暂缓：HF card 和 vLLM 文档确认 MIT 与后端支持，但该任务优先中文古诗词理解，第一轮资源应给中文模型和 formatter 主候选。来源：[Phi-4](https://huggingface.co/microsoft/phi-4)、[Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)、[vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/)。
8. GPT-OSS-20B 暂缓：OpenAI 官方和 HF card 确认 Apache-2.0 与开放权重，vLLM 支持 `GptOssForCausalLM`；但严格总参数小于 20B 时不适合第一轮 harness。来源：[OpenAI gpt-oss blog](https://openai.com/index/introducing-gpt-oss)、[gpt-oss-20b card](https://huggingface.co/openai/gpt-oss-20b)、[vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/)。
