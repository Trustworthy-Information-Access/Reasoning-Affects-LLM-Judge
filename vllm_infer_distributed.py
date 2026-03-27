import json 
from vllm import LLM, SamplingParams
from typing import Any, Dict, List
import ray
import numpy as np
from packaging.version import Version
import argparse
import sys
import os
sys.path.append('..')
from prompts.convert import MyDataset, read_json
from transformers import AutoTokenizer, AutoConfig

assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"

def deepseek_r1_distill_qwen_template(instruction, think='False'):
    if think == 'False':
        data = f"<|begin▁of▁sentence|><|User|>{instruction}<|Assistant|><think>So I need to give certain if I can provide the correct answer. Otherwise, I should say uncertain.\n</think>"
    else:
        data = f"<|begin▁of▁sentence|><|User|>{instruction}<|Assistant|>"
    return data

def qwen2_qwen3_instruct_template(instruction, template, think='False'):
    if think == 'False':
        template_noinput = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n<think></think>'
    else:
        template_noinput = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n'
    # 对qwen2来说, 无论think是什么都不该出现<think>
    if template == 'qwen2':
        template_noinput = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n'
    data = template_noinput.format_map({"instruction": instruction})
    return data

def glm4_instruct_template(instruction, think='False'):
    if think == 'False':
        template_noinput = '[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n{instruction}<|assistant|>\n<think></think>'
    else:
        template_noinput = '[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n{instruction}<|assistant|>\n<think>'
    data = template_noinput.format_map({"instruction": instruction})
    return data

def llama2_instruct_template(instruction, think='False'):
    template_noinput = '<s>[INST] <<SYS>>\nYou are a helpful assistant<</SYS>>\n\n{instruction}[/INST]'
    data = template_noinput.format_map({"instruction": instruction})
    return data

def llama3_instruct_template(instruction, think='False'):
    template_noinput = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    data = template_noinput.format_map({"instruction": instruction})
    return data

def infer(template, input_data, output_path, model_path, tp_size, temperature, topp, topk, max_tokens, repetition_penalty, sample_num, num_instances, batchsize, logprobs, get_tokens, think='False'):
    sampling_params = SamplingParams(n=sample_num, temperature=temperature, top_k=topk, top_p=topp, max_tokens=max_tokens, repetition_penalty=repetition_penalty, logprobs=logprobs, stop=["<|end▁of▁sentence|>", "<|eot_id|>"])

    class LLMPredictor:
        def __init__(self):
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # 获取模型配置
            config = AutoConfig.from_pretrained(model_path)

            # 读取最大支持的 context 长度
            max_supported_len = getattr(config, "max_position_embeddings", tokenizer.model_max_length)
            max_model_len = min(max_supported_len, 10000)
            self.llm = LLM(model=model_path, tensor_parallel_size=tp_size, max_model_len=max_model_len)
            

        def __call__(self, batch: Dict[str, Any]) -> Dict[str, list]:
            outputs = self.llm.generate(batch["instruction"], sampling_params)
            results = []
            for idx, output in zip(batch["index"], outputs):
                temp_res = {
                    "index": idx,
                    "prompt": output.prompt,
                    "response": [o.text for o in output.outputs],
                }
                if get_tokens:
                    temp_res['tokens'] = [o.token_ids for o in output.outputs]
                if logprobs:
                    temp_res['logprobs'] = [o.logprobs for o in output.outputs]
                    temp_res['cumulative_logprobs'] = [o.cumulative_logprob for o in output.outputs]

                results.append(temp_res)
            return {"results": results}

    # 添加索引
    all_prompts = []
    for i, sample in enumerate(input_data):
        prompt = sample['instruction']
        if template == 'qwen2' or template == 'qwen3':
            formatted_prompt = qwen2_qwen3_instruct_template(prompt, template, think)
        elif template == 'llama2':
            formatted_prompt = llama2_instruct_template(prompt, think)
        elif template == 'llama3':
            formatted_prompt = llama3_instruct_template(prompt, think)
        elif template == 'deepseek_r1_distill_qwen':
            formatted_prompt = deepseek_r1_distill_qwen_template(prompt, think)
        elif template == 'glm':
            formatted_prompt = glm4_instruct_template(prompt, think)
        else:
            raise ValueError(f'template not supported: {template}')
        all_prompts.append({"index": i, "instruction": formatted_prompt})

    ds = ray.data.from_items(all_prompts)
    resources_kwarg: Dict[str, Any] = {"num_gpus": tp_size}
    total_items = len(all_prompts)

    if os.path.exists(output_path):
        check_out_data = read_json(output_path)
        if len(check_out_data) == total_items:
            print(f'output file already exists, skip inference')
            return

    # 尽量保证数据批次为gpus个数的倍数
    cur_batchs = 8
    cur_batch_size = total_items // cur_batchs  # 初始尝试将数据分成8份
    max_batch_size=batchsize
    while cur_batch_size > max_batch_size:
        if cur_batch_size <= 128:
            break
        cur_batchs *= 2
        cur_batch_size = total_items // cur_batchs
    print(f'final batch_size: {cur_batch_size}')

    ds = ds.map_batches(
        LLMPredictor,
        concurrency=num_instances,
        batch_size=cur_batch_size,
        **resources_kwarg,
    )

    outputs = ds.take_all()
    outputs = [item['results'] for item in outputs]

    # **按照 index 排序**
    sorted_outputs = sorted(outputs, key=lambda x: x["index"])

    outdir = os.path.dirname(output_path)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(output_path, 'w') as fw:
        for idx, output in enumerate(sorted_outputs):
            line = {
                'question': input_data[idx]['question'],    
                'prompt': output["prompt"],
                'answer': input_data[idx]['answer'],
                'response': output["response"],
            }
            if get_tokens:
                line['tokens'] = output['tokens']
            if logprobs >= 1:
                line['cumulative_logprobs'] = output['cumulative_logprobs']
                logprobs_res = []
                for sample_logprobs in output['logprobs']:
                    sample_res = []
                    for item in sample_logprobs:
                        item_res = {}
                        for k, v in item.items(): 
                            item_res[k] = {'logprob': v.logprob, 'rank': v.rank, 'decoded_token': v.decoded_token}
                        sample_res.append(item_res)
                    logprobs_res.append(sample_res)
                line['logprobs'] = logprobs_res
                
            fw.write(json.dumps(line, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vllm distributed batch inference')
    parser.add_argument('--template', type=str, required=True, help='chat template.')
    parser.add_argument('--model_path', type=str, required=True, help='path to your model.')
    parser.add_argument('--tp_size', type=int, required=True, help='tensor parallel size. e.g. 8 for 72B models, 1 for 7B models.')
    parser.add_argument('--input_path', type=str, required=True, help='input jsonl path.')
    parser.add_argument('--output_path', type=str, required=True, help='output jsonl path.')
    parser.add_argument('--temperature', type=float, required=True, help='sample temperature')
    parser.add_argument('--topp', type=float, required=True, help='top p value.')
    parser.add_argument('--topk', type=int, required=True, help='top k value.')
    parser.add_argument('--max_tokens', type=int, required=True, help='max token length.')
    parser.add_argument('--repetition_penalty', type=float, required=True, help='repetition penalty value.')
    parser.add_argument('--sample_num', type=int, required=True, help='sample beam size.')
    parser.add_argument('--num_instances', type=int, required=True, help='num of distributed instances.')
    parser.add_argument('--batchsize', type=int, required=True, help='infer batch size.')
    parser.add_argument('--prompt_type', type=str, required=True, default='qa')
    parser.add_argument('--logprobs', type=int, required=True, default=0)
    parser.add_argument('--get_tokens', type=int, required=True, default=0, choices=[0, 1])
    parser.add_argument('--dataset_name', type=str, required=True, default='nq')
    parser.add_argument('--think', type=str, required=True, default='False', choices=['True', 'False'])

    args = parser.parse_args()
    print(f'think: {args.think}')
    data = MyDataset(args.input_path, args.dataset_name)
    print(len(data))
    input_data = data.prepare_prompts(args.prompt_type)
    print(f'len data: {len(input_data)}')
    show_cnt = 0
    for item in input_data:
        print(item['instruction'])
        show_cnt += 1
        if show_cnt == 5:
            break
    infer(args.template,
          input_data, 
          args.output_path, 
          args.model_path, 
          args.tp_size, 
          args.temperature, 
          args.topp, 
          args.topk, 
          args.max_tokens, 
          args.repetition_penalty, 
          args.sample_num, 
          args.num_instances, 
          args.batchsize,
          args.logprobs,
          args.get_tokens,
          args.think)
