import json
from torch.utils.data import DataLoader, Dataset, RandomSampler
from utils.prompt import get_prompt
import pandas as pd
import os
import random

def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

def assign_levels(data, field, num_levels=10):
    """
    根据指定字段对数据分级。
    :param data: 输入数据列表，每个数据是一个字典
    :param field: 用于分级的字段名
    :param num_levels: 分级数
    :return: 每条数据的等级列表
    """
    # 提取字段值并排序（过滤掉非数值型字段）
    field_values = [item[field] for item in data if isinstance(item[field], (int, float))]
    unique_values = sorted(set(field_values))
    
    # 计算分级边界
    step = len(unique_values) // num_levels
    boundaries = [unique_values[i * step] for i in range(1, num_levels)]
    boundaries.append(float('inf'))
    
    # 为每条数据分配等级
    levels = []
    for item in data:
        value = item[field]
        if isinstance(value, (int, float)):
            current_level = 1
            for boundary in boundaries:
                if value <= boundary:
                    break
                current_level += 1
            levels.append(current_level)
            item[field + '_level'] = current_level
        else:
            levels.append(None)  # 非数值型字段分配为 None
    return levels

def select_samples_from_levels(data, levels, required_levels):
    """
    从指定的等级中各取一个样本。
    :param data: 输入数据列表
    :param levels: 每条数据对应的等级
    :param required_levels: 需要的等级列表
    :return: few-shot 样本列表
    """
    # 按等级分组
    grouped_by_level = {}
    for item, level in zip(data, levels):
        if level is not None:
            grouped_by_level.setdefault(level, []).append(item)
    
    # 从指定的等级中各取一个样本
    selected_samples = []
    for level in required_levels:
        if level in grouped_by_level and grouped_by_level[level]:
            selected_samples.append(grouped_by_level[level].pop(0))  # 从当前等级取一个样本
    return selected_samples



class QADataset(Dataset):
    """
    Open-domain generation dataset
    """
    def __init__(self, args):
        self.data = self.read(args.source)
        self.prompts = []
        self.idxs = []
        self.args = args
        self.few_shot_examples = []
        if args.n_shot > 0:
            self.few_shot_examples = self.get_few_shot_examples()
        self.get_prompted_data()

    def read(self, path):
        qa_data = []
        f = open(path, 'r', encoding='utf-8')
        for line in f.readlines():
            qa_data.append(json.loads(line))
        return qa_data
    
    def get_prompted_data(self):
        for idx in range(len(self.data)):
            if 'info' not in self.data[idx]:
                self.idxs.append(idx)
                self.prompts.append(get_prompt(self.data[idx], self.args, self.few_shot_examples)) 
        for item in self.prompts[:5]:
            print(f'example: {item}')

    def get_few_shot_examples(self):
        use_model = 'llama8b'
        data = read_json(f'./res/clean_data_for_pop_generation/{self.args.dataset_name}_{use_model}_temperature1.jsonl')
        new_data = []
        for item in data:
            if item["question_pop"] != "No" and item["gene_pop"] != "No":
                new_data.append(item)

        field_to_rank = f"{self.args.gene_type}_pop"  # 可修改为 "coo_pop" 或其他字段
        levels = assign_levels(data, field_to_rank, num_levels=10)

        # 分别取 3、5、10 个样本
        if self.args.n_shot == 3:
            need_levels = [2, 5, 8]
        elif self.args.n_shot == 5:
            need_levels = [1, 3, 5, 7, 9]
        elif self.args.n_shot == 10:
            need_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        few_shot_examples = select_samples_from_levels(data, levels, need_levels)

        return few_shot_examples

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.prompts[index]
    
class MCDataset(Dataset):
    """
    Multi-choice dataset
    """
    # generate input for the given subject
    def __init__(self, args, subject):
        self.args = args
        self.subject = subject
        self.data = self.read('test')
        self.idxs = range(len(self.data))
        self.dev_data = self.read('dev') if self.args.n_shot != 0 else []
        self.get_choice_count()
        self.prompts = []
        self.get_prompted_data()
    
    def get_choice_count(self):
        all_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.choice_cnt = len(self.data[0]) - 2
        self.choices = all_choices[:self.choice_cnt]


    def read(self, mode='test'):
        mmlu_data = pd.read_csv(os.path.join(self.args.source, self.args.data_mode, self.subject + f"_{mode}.csv"), header=None).to_numpy() # no header
        return mmlu_data
    
    def format_subject(self, subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s
    
    def format_example(self, data, idx, include_answer=True):
        prompt = data[idx][0] # question
        k = len(data[idx]) - 2 # count of choices
        for j in range(k):
            prompt += "\n{}. {}".format(self.choices[j], data[idx][j+1]) # append each candidate answer
        return prompt
    
    def get_prompted_data(self):
        if self.args.task == 'mmlu':
            self.args.subject = ' about' + self.format_subject(self.subject) 
        else:
            self.args.subject = ''
        for idx in range(len(self.data)):
            question = self.format_example(self.data, idx, include_answer=False)
            prompt = get_prompt({'question': question}, self.args)
            self.prompts.append(prompt)
        for item in self.prompts[:5]:
            print(f'example: {item}')
        prompt_len = []
        for item in self.prompts:
            prompt_len.append(len(item.split(' ')))
        self.avg_len = sum(prompt_len)/len(prompt_len)
        self.max_len = max(prompt_len)

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.prompts[index]
    





