# Baseline Smoke Results

Generated at: 2026-05-21 20:41:39 +0800

Scope: 1-2 sample smoke for prompt baseline and formatter links. No LLM judge scores were computed.

Decode params: `temperature=0`, `top_p=0.8`, `max_tokens=768`. Qwen3 requests sent `chat_template_kwargs.enable_thinking=false` and appended `/no_think` as a fallback.

## Result Table

| experiment_id | group | model | backend | quantization | prompt_type | shot_count | sample_count | json_error_rate | avg_latency_ms | p95_latency_ms | status | notes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| P14-qwen3-14b-bf16-vllm-nothink-zero | P14 | Qwen/Qwen3-14B | vllm | bf16 | zero-shot | 0 | 1 | 0.0 | 5081.6 | 5081.6 | ok |  |
| P14-fast-qwen3-14b-awq4-vllm-nothink-zero | P14-fast | Qwen/Qwen3-14B-AWQ | vllm | awq4 | zero-shot | 0 | 1 | 0.0 | 17989.17 | 17989.17 | ok |  |
| P8-qwen3-8b-bf16-vllm-nothink-zero | P8 | Qwen/Qwen3-8B | vllm | bf16 | zero-shot | 0 | 2 | 0.5 | 1499.22 | 1739.56 | ok |  |
| P8-qwen3-8b-awq4-vllm-nothink-zero | P8 | Qwen/Qwen3-8B-AWQ | vllm | awq4 | zero-shot | 0 | 2 | 0.5 | 6051.45 | 6574.04 | ok |  |
| P8-internlm3-8b-instruct-bf16-vllm-normal-zero | P8 | internlm/internlm3-8b-instruct | vllm | bf16 | zero-shot | 0 | 2 | 1.0 | 2478.39 | 2626.74 | ok |  |
| FMT-qwen3-8b-bf16-vllm-nothink-jsonfix | FMT | Qwen/Qwen3-8B | vllm | bf16 | fmt | 0 | 2 | 0.0 | 1925.1 | 2137.33 | ok |  |
| FMT-gemma-4-e4b-it-bf16-vllm-direct-json-jsonfix | FMT | google/gemma-4-E4B-it | vllm | bf16 | fmt | 0 | 2 | 0.5 | 1528.77 | 2316.79 | ok |  |

## JSON Error Samples

| idx | experiment_id | raw_output excerpt | error categories | validation_error |
| --- | --- | --- | --- | --- |
| 0 | P8-qwen3-8b-bf16-vllm-nothink-zero | {"idx":0,"ans_qa_words":{"衰草":"荒芜的草","故关":"故乡的关隘","风尘":"旅途的尘土"},"ans_qa_sents":{"故关衰草遍，离别自堪悲":"故乡的草木一片荒芜，离别的场景本身就令人悲伤","掩泪空相向，风尘何处期":"只能含泪空对，不知何时才能在风尘中重逢"},"... | missing_sentence_key, extra_sentence_key | {   "valid": false,   "errors": [     "missing_ans_qa_sents",     "unexpected_ans_qa_sents"   ],   "warnings": [     "empty_ans_qa_sents"   ],   "normalized": {     "idx": 0,     "ans_qa_words": {       "衰草": "荒芜的草",       "故关": "故乡的关隘",       "风尘": "旅途的尘土"     },     "ans_qa_sents": {       "故关衰草遍，离别自堪悲": "故乡的草木一片荒芜，离别的场景本身就令人悲伤",       "掩泪空相向，风尘何处期": "只能含泪空对，不知何时才能在风尘中重逢"     },     "choose_id": "D"   } } |
| 0 | P8-qwen3-8b-awq4-vllm-nothink-zero | {   "idx": 0,   "ans_qa_words": {     "衰草": "荒凉的草地",     "故关": "故乡的关隘",     "风尘": "旅途的尘土"   },   "ans_qa_sents": {     "故关衰草遍，离别自堪悲": "故乡的草木一片荒凉，离别本身就令人悲伤", ... | missing_sentence_key, extra_sentence_key | {   "valid": false,   "errors": [     "missing_ans_qa_sents",     "unexpected_ans_qa_sents"   ],   "warnings": [     "empty_ans_qa_sents"   ],   "normalized": {     "idx": 0,     "ans_qa_words": {       "衰草": "荒凉的草地",       "故关": "故乡的关隘",       "风尘": "旅途的尘土"     },     "ans_qa_sents": {       "故关衰草遍，离别自堪悲": "故乡的草木一片荒凉，离别本身就令人悲伤",       "掩泪空相向，风尘何处期": "只能含泪遥望，不知何时才能重逢"     },     "choose_id": "D"   } } |
| 0 | P8-internlm3-8b-instruct-bf16-vllm-normal-zero | ```json {   "idx": 0,   "ans_qa_words": {     "衰草": "枯萎的草",     "故关": "故乡",     "风尘": "尘世的纷扰"   },   "ans_qa_sents": {     "故关衰草遍，离别自堪悲": "故乡遍地枯草，离别时自然令人悲伤",... | extra_text, missing_sentence_key, extra_sentence_key | {   "valid": false,   "errors": [     "missing_ans_qa_sents",     "unexpected_ans_qa_sents"   ],   "warnings": [     "empty_ans_qa_sents"   ],   "normalized": {     "idx": 0,     "ans_qa_words": {       "衰草": "枯萎的草",       "故关": "故乡",       "风尘": "尘世的纷扰"     },     "ans_qa_sents": {       "故关衰草遍，离别自堪悲": "故乡遍地枯草，离别时自然令人悲伤",       "掩泪空相向，风尘何处期": "掩面而泣，不知何时能再相见"     },     "choose_id": "D"   } } |
| 1 | P8-internlm3-8b-instruct-bf16-vllm-normal-zero | ```json {   "idx": 1,   "ans_qa_words": {     "窥": "偷偷地窥视",     "牧马": "放牧的马匹",     "临洮": "地名，古代边境要塞"   },   "ans_qa_sents": {     "至今窥牧马": "直到现在，敌人还在偷偷地窥视放牧的... | extra_text |  |
| 1 | FMT-gemma-4-e4b-it-bf16-vllm-direct-json-jsonfix | {"idx": 1, "ans_qa_words": {"窥": "偷看，窥探", "牧马": "", "临洮": ""}, "ans_qa_sents": {"至今窥牧马": "", "不敢过临洮": ""}, "choose_id": ""} | empty_required_answer, invalid_choose_id | {   "valid": false,   "errors": [     "invalid_choose_id"   ],   "warnings": [     "empty_ans_qa_words",     "empty_ans_qa_sents"   ],   "normalized": {     "idx": 1,     "ans_qa_words": {       "窥": "偷看，窥探",       "牧马": "",       "临洮": ""     },     "ans_qa_sents": {       "至今窥牧马": "",       "不敢过临洮": ""     },     "choose_id": ""   } } |

## Latency Records

Per-sample details: `data/baseline/smoke-results.jsonl`.

| experiment_id | avg_latency_ms | p95_latency_ms | sample_count |
| --- | ---: | ---: | ---: |
| P14-qwen3-14b-bf16-vllm-nothink-zero | 5081.6 | 5081.6 | 1 |
| P14-fast-qwen3-14b-awq4-vllm-nothink-zero | 17989.17 | 17989.17 | 1 |
| P8-qwen3-8b-bf16-vllm-nothink-zero | 1499.22 | 1739.56 | 2 |
| P8-qwen3-8b-awq4-vllm-nothink-zero | 6051.45 | 6574.04 | 2 |
| P8-internlm3-8b-instruct-bf16-vllm-normal-zero | 2478.39 | 2626.74 | 2 |
| FMT-qwen3-8b-bf16-vllm-nothink-jsonfix | 1925.1 | 2137.33 | 2 |
| FMT-gemma-4-e4b-it-bf16-vllm-direct-json-jsonfix | 1528.77 | 2316.79 | 2 |

## Run Failures

No model startup or generation failures were recorded.

## vLLM Cleanup

This run cleaned up all vLLM processes it started: `True`.

Machine-readable summary: `data/baseline/smoke-summary.json`.
Failure details: `data/baseline/smoke-failures.jsonl`.
