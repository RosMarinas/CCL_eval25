# E3 dev-50 Baseline Results

Generated at: 2026-05-23 02:48:48 +0800

Scope: 50 samples (first 50 from eval_data.json), 8 prompt-only experiments.
No LLM judge scores — JSON stability and latency only.

Decode params: temperature=0, top_p=0.8, max_tokens=2048 (think) / 768 (nothink).

## Result Table

| # | experiment_id | thinking | n | json_err | hard_err | fmt_err | avg_lat_ms | p95_lat_ms | status |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | P14-qwen3-14b-bf16-vllm-nothink-zero | non-thinking | 50 | 0.000 | 0.000 | 0.000 | 4564.44 | 5725.5 | ok |
| 2 | P14-qwen3-14b-bf16-vllm-think-zero | thinking | 50 | 0.000 | 0.000 | 1.000 | 18089.02 | 24687.7 | ok |
| 3 | P8-qwen3-8b-bf16-vllm-nothink-zero | non-thinking | 50 | 0.040 | 0.040 | 0.040 | 1878.51 | 2485.61 | ok |
| 4 | P8-qwen3-8b-bf16-vllm-think-zero | thinking | 50 | 0.040 | 0.020 | 0.980 | 11375.98 | 16769.89 | ok |
| 5 | P8-qwen3-8b-awq4-vllm-nothink-zero | non-thinking | 50 | 0.000 | 0.000 | 0.020 | 7303.32 | 9469.55 | ok |
| 6 | P8-internlm3-8b-instruct-bf16-vllm-think-zero | thinking | 50 | 0.060 | 0.000 | 1.000 | 3020.77 | 3886.19 | ok |
| 7 | P14-fast-qwen3-14b-awq4-vllm-nothink-zero | non-thinking | 50 | 0.000 | 0.000 | 0.000 | 17423.25 | 22089.54 | ok |
| 8 | P8-internlm3-8b-instruct-bf16-vllm-normal-zero | normal | 50 | 0.060 | 0.000 | 1.000 | 3020.77 | 3885.52 | ok |

## Thinking vs Nothink Ablation

| pair | mode | json_err | hard_err | fmt_err | avg_lat_ms | p95_lat_ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P14 Qwen3-14B bf16 | think | 0.000 | 0.000 | 1.000 | 18089.02 | 24687.7 |
| P14 Qwen3-14B bf16 | nothink | 0.000 | 0.000 | 0.000 | 4564.44 | 5725.5 |
| P8 Qwen3-8B bf16 | think | 0.040 | 0.020 | 0.980 | 11375.98 | 16769.89 |
| P8 Qwen3-8B bf16 | nothink | 0.040 | 0.040 | 0.040 | 1878.51 | 2485.61 |

InternLM3-8B:

| mode | json_err | hard_err | fmt_err | avg_lat_ms | p95_lat_ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| thinking | 0.060 | 0.000 | 1.000 | 3020.77 | 3886.19 |
| normal | 0.060 | 0.000 | 1.000 | 3020.77 | 3885.52 |

## JSON Error Samples

| idx | experiment_id | error categories | raw_output excerpt |
| ---: | --- | --- | --- |
| 0 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> </think>  {   "idx": 0,   "ans_qa_words": {     "衰草": "形容荒凉的景象，象征离别时的凄凉。",     "故关": "故乡的关隘，指离别的地方。",     "风尘": "比喻旅途的艰辛和漂泊不定。"   },   "ans_qa_sents": {     "故关衰草遍，离别自堪悲": "故乡的关隘长满衰草，离别之情自然... |
| 1 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是《哥舒歌》由西鄙人所作，内容为：“北斗七星高，哥舒夜带刀。至今窥牧马，不敢过临洮。” 任务包括解释特定词语、翻译特定句子，并选择最符合全诗情感的选项。  首先处理ans_qa_words部分。用户提供的qa_words有“窥”、“牧马”、“临洮”。需要逐个解释... |
| 2 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是张泌的《寄人》，内容是四句诗。任务包括解释特定词语、翻译特定句子，并选择最符合全诗情感的选项。用户要求输出一个JSON对象，包含idx、ans_qa_words、ans_qa_sents和choose_id四个字段，且必须严格按照格式来。  首先处理词语解释部... |
| 3 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是杜甫的《闻官军收河南河北》，内容是关于听到官军收复失地后的喜悦心情。用户要求根据输入的诗歌、目标词语、目标句子和情感选项，生成一个符合特定格式的JSON答案。  首先，处理idx字段，必须与输入的idx完全一致，这里输入的idx是3，所以输出的JSON中idx... |
| 4 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  用户给出的诗歌是卢纶的《晚次鄂州》，需要处理三个问题：词语解释、句子翻译以及情感选择。首先，我需要确认输入的各个字段，包括idx、title、author、content、qa_words、qa_sents和choose选项。  对于ans_qa_words部分，需要处理... |
| 5 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是顾况的《宫词》，内容为四句。任务包括解释特定词语、翻译特定句子，并选择最符合全诗情感的选项。用户要求输出一个JSON对象，包含idx、ans_qa_words、ans_qa_sents和choose_id四个字段，且必须严格按照格式要求，不能有任何多余内容。 ... |
| 6 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是杜甫的《天末怀李白》，内容是表达对李白的思念和担忧。用户要求根据输入的诗歌、目标词语、目标句子和情感选项生成一个JSON答案。输出必须严格遵循格式，只包含指定的字段，并且不能有任何额外内容。  首先处理idx，确保与输入一致，这里是6。接下来是ans_qa_w... |
| 7 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是温庭筠的《送人东游》，内容包括标题、作者、诗句、需要解释的词语和句子，以及情感选项。用户要求生成一个JSON对象，包含idx、ans_qa_words、ans_qa_sents和choose_id四个字段。  首先处理ans_qa_words部分。用户提供的q... |
| 8 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是杜牧的《泊秦淮》，内容是“烟笼寒水月笼沙，夜泊秦淮近酒家。商女不知亡国恨，隔江犹唱后庭花。”任务包括解释特定词语、翻译特定句子，并选择最符合全诗情感的选项。  首先处理词语解释部分。qa_words中有三个词：“泊”、“商女”、“后庭花”。对于“泊”，在诗中是... |
| 9 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是刘禹锡的《乌衣巷》，内容是四句诗。任务包括解释特定词语、翻译特定句子，并选择最符合全诗情感的选项。用户要求输出一个JSON对象，包含idx、ans_qa_words、ans_qa_sents和choose_id四个字段，且必须严格按照格式来。  首先处理词语解... |
| 10 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是张旭的《桃花溪》，内容是四句诗。任务要求根据输入的诗歌、目标词语、目标句子和情感选项，生成一个符合特定格式的JSON答案。用户特别强调只输出JSON，不要任何其他内容，所以必须严格遵守格式要求。  首先处理idx字段，确保与输入一致，这里应该是10。接下来是a... |
| 11 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是贾岛的《寻隐者不遇》，内容是四句诗。任务包括解释特定词语、翻译特定句子，并选择最符合全诗情感的选项。用户要求输出一个JSON对象，包含idx、ans_qa_words、ans_qa_sents和choose_id四个字段，且必须严格按照格式来。  首先处理an... |
| 12 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是杜甫的《宿府》，用户需要根据提供的qa_words和qa_sents来生成对应的解释，并选择最符合全诗情感的选项。首先，我需要处理ans_qa_words部分。给出的词语是“井梧”、“中天”、“伶俜”。对于“井梧”，我需要解释它在诗中的意思，可能指井边的梧桐树... |
| 13 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是张祜的《集灵台·其二》，内容是四句诗。任务是根据给定的qa_words、qa_sents和情感选项生成对应的JSON答案。用户要求直接输出最终的JSON，不需要任何解释或额外内容，所以我要确保严格按照格式来。  首先处理ans_qa_words部分。输入的qa... |
| 14 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读用户的要求，确保完全理解每一个细节。用户给了一个JSON输入，包含诗歌内容、问题词语、问题句子以及情感选项，要求生成符合特定格式的JSON输出。  首先，我需要确认输入中的各个部分。诗歌是李颀的《送魏万之京》，内容有八句。用户需要处理的词语是“蹉跎”、“离歌”、“砧声”，句子是前两句和最后两句。情感选项是四个：A... |
| 15 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是杜甫的作品，题目比较长，内容有四句。用户需要根据提供的qa_words和qa_sents来生成对应的解释，并选择最符合全诗情感的选项。  首先处理ans_qa_words部分。qa_words里有三个词：“间道”、“破胆”、“移官”。我需要逐个解释它们在诗中的... |
| 16 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是张祜的《集灵台·其一》，内容有四句。用户要求根据输入的诗歌、目标词语、目标句子和情感选项生成一个JSON答案。输出必须严格遵循格式，只包含指定的字段，并且不能有任何额外内容。  首先处理idx，必须与输入一致，这里输入的idx是16，所以直接保留。接下来是an... |
| 17 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是韩偓的《已凉》，内容有四句。用户要求根据输入的诗歌、目标词语、目标句子和情感选项生成一个JSON答案。输出必须严格符合格式，只包含指定的字段，并且不能有任何额外内容。  首先处理idx，必须与输入一致，这里输入的idx是17，所以输出的JSON中idx应该是1... |
| 18 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是杜甫的《奉济驿重送严公四韵》，内容是送别诗，表达了对友人离别的不舍和自身孤独的情感。用户要求根据输入的诗歌、目标词语、目标句子和情感选项，生成符合要求的JSON答案。  首先处理ans_qa_words部分。用户提供的qa_words包括“列郡”、“三朝”、“... |
| 19 | P14-qwen3-14b-bf16-vllm-think-zero | extra_text, thinking_trace_leak | <think> 好的，我现在需要处理用户提供的古诗词理解任务。首先，我要仔细阅读输入内容，确保正确理解每个部分的要求。  输入的诗歌是刘长卿的《新年作》，内容有八句。用户要求根据输入的诗歌、目标词语、目标句子和情感选项生成一个JSON答案。输出必须严格符合格式，只包含指定的字段，并且不能有任何额外内容。  首先处理idx，必须与输入一致，这里输入的idx是19，所以直接保留。接下来是ans_... |
| ... | (183 more) | | |

## Latency Records

| experiment_id | avg_latency_ms | p95_latency_ms | n |
| --- | ---: | ---: | ---: |
| P14-qwen3-14b-bf16-vllm-nothink-zero | 4564.44 | 5725.5 | 50 |
| P14-qwen3-14b-bf16-vllm-think-zero | 18089.02 | 24687.7 | 50 |
| P8-qwen3-8b-bf16-vllm-nothink-zero | 1878.51 | 2485.61 | 50 |
| P8-qwen3-8b-bf16-vllm-think-zero | 11375.98 | 16769.89 | 50 |
| P8-qwen3-8b-awq4-vllm-nothink-zero | 7303.32 | 9469.55 | 50 |
| P8-internlm3-8b-instruct-bf16-vllm-think-zero | 3020.77 | 3886.19 | 50 |
| P14-fast-qwen3-14b-awq4-vllm-nothink-zero | 17423.25 | 22089.54 | 50 |
| P8-internlm3-8b-instruct-bf16-vllm-normal-zero | 3020.77 | 3885.52 | 50 |

Per-model data: `data/baseline/e3-dev50/<experiment_id>.jsonl`
