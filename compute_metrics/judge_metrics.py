import os
import json
import csv
import re
from utils import has_answer


def deal_judge_new(pred):
    if pred is None:
        return True
    return has_answer([
        "sorry", "apologize", "apologies", "uncertain", "false", "no", 'unsure',
        "cannot", "unknown", "no specific answer", "not provide", "cannot answer",
        "no information provided", "no answer", "not contain", "no definitive answer"
    ], pred)

#每条json数据有对应flag（true/false）
def read_flags(path):
    flags = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            resp = data['response'][0].strip().lower()
            flags.append(deal_judge_new(resp))
    return flags


# 数据路径
data_dir = ''
csv_path = ''


fieldnames = [
    'filename',
    'judge_model', 'prompt_type',
    'temperature', 'top_p', 'top_k',
    'think_used_in_qa', 'think_used_in_judge',
    'alignment',   # judge vs gold 一致率
    'confidence',           # certain 比例
    'overconfidence',  # judge certain but gold uncertain
    'conservative'     # judge uncertain but gold certain
]

# 正则匹配
filename_pattern = re.compile(
    r'^MATH500_'                                      
    r'(?P<qa_model>Qwen3-[0-9]+B)_'              
    r'(?P<prompt_type>short_qa|long_qa)_'        
    r'(?P<temperature>[0-9\.]+)_'                
    r'(?P<topp>[0-9\.]+)_'                       
    r'(?P<topk>[0-9]+)_think_'                   
    r'(?P<think_qa>True|False)_sample_1_'       
    r'judge_llm_judge'                           
    r'(?:_with_think)?_'                         
    r'(?P<judge_model>.+?)_'                     
    r'think_(?P<think_judge>True|False)\.jsonl$' 
)

records = []

#遍历所有judge文件
for filename in os.listdir(data_dir):
    if not filename.endswith('.jsonl') or 'judge_llm_judge' not in filename:
        continue

    m = filename_pattern.match(filename)
    if not m:
        print(f"跳过无法解析的文件名：{filename}")
        continue

    flds = m.groupdict()
    fp_judge = os.path.join(data_dir, filename)

    # 读取 judge flags
    judge_flags = read_flags(fp_judge)
    total = len(judge_flags)
    certain_count = total - sum(judge_flags)
    confidence = round(certain_count / total, 4) if total else 0.0

    # 对应 gold 文件
    gold_fn = (
        f"AIME24_{flds['qa_model']}_{flds['prompt_type']}_"
        f"{flds['temperature']}_{flds['topp']}_{flds['topk']}_"
        f"think_{flds['think_qa']}_sample_1_judge_gold_Qwen2.5-72B-Instruct.jsonl"
    )
    
    fp_gold = os.path.join(data_dir, gold_fn)
    if not os.path.exists(fp_gold):
        print(f"未找到对应的 Gold 文件：")
        continue

    # 读取 gold flags
    gold_flags = read_flags(fp_gold)
    if len(gold_flags) != total:
        print(f"行数不一致：{filename} vs {gold_fn}")
        continue

    match_count = sum(g == j for g, j in zip(gold_flags, judge_flags))
    alignment = round(match_count / total, 4)

    # overconfidence: gold=True (uncertain) & judge=False (certain)
    overconf_count = sum(g and not j for g, j in zip(gold_flags, judge_flags))
    overconfidence  = round(overconf_count / total, 4)

    # conservative: gold=False (certain) & judge=True (uncertain)
    cons_count   = sum((not g) and j for g, j in zip(gold_flags, judge_flags))
    conservative    = round(cons_count / total, 4)


    records.append({
        'filename'             : filename,
        # 'qa_model'             : flds['qa_model'],
        'judge_model'          : flds['judge_model'],
        'prompt_type'          : flds['prompt_type'],
        'temperature'          : flds['temperature'],
        'top_p'                : flds['topp'],
        'top_k'                : flds['topk'],
        'think_used_in_qa'     : flds['think_qa'],
        'think_used_in_judge'  : flds['think_judge'],
        'alignment'         : alignment,
        'confidence'           : confidence,
        'overconfidence'  : overconfidence,
        'conservative'    : conservative
    })

with open(csv_path, 'w', encoding='utf-8', newline='') as wf:
    writer = csv.DictWriter(wf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"已保存 {len(records)} 条记录到 {csv_path}")
