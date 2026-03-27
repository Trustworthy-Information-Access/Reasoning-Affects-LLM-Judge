import json
from typing import Any
import re
# 这里的convert为了配合deepseek，修改了short_qa和llm_judge_with_think！

prompt_dict = {
    'short_qa': {
        'none': 'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}',
        'tail': '\nAnswer: ',
    },
    'short_mc_qa': {
        'none': 'Select the correct answer to the following question based on your internal knowledge. Only give your choice(A/B/C/D) without any other words.\nQuestion: {question}',
        'tail': '\nAnswer: ',
    },
    'long_mc_qa': {
        'none': 'Select the correct answer to the following question based on your internal knowledge.\nQuestion: {question}',
        'tail': '\nAnswer: ',
    },
    'long_qa': {
        'none': 'Answer the following question.\nQuestion: {question}',
        'tail': '\nAnswer: ',
    },
    'qa_conf':{
        'none': 'Answer the following question based on your internal knowledge with one or few words. If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\"\nQuestion: {question}',
        'tail': '\nAnswer: ',
    },
    'conf_qa':{
        'none': 'Answer the following question based on your internal knowledge with one or few words. If you are sure your answer is correct, give the answer. If you can not give the correct answer, please say \"uncertain\"\nQuestion: {question}',
        'tail': '\nAnswer: ',
    },
    'prior_conf':{
        'none': 'Judge whether you can provide a correct answer for the given question. If you are sure you can give a correct answer, say \"certain\". If not, please say \"uncertain\". Just give your judgement without any other words.\nQuestion: {question}',
        'tail': '\nAnswer: ',
    },
    'post_conf':{
        'none': 'Judge whether the following answer (this is your self-generated answer) about the question is correct. If you are sure the answer is correct, say \"certain\". If not, please say \"uncertain\". Just give your judgement without any other words.\nQuestion: {question}.\nAnswer:{prediction}',
        'tail': '\nAnswer: ',
    },
    'consistency':{
        'none': '{question}.',
        'tail': '\nAnswer: ',
    },
    'llm_judge_gold':{
        'none': 'We are assessing the quality of answers to the following question: {question}.\nThe expected answers to this question are: {answer}.\nWithin the context of the question, dose the generated answer mean the same as any of the expected answers?\nThe generated answer is: {prediction}.\nIf the answer is correct, say \"certain\". If not, please say \"uncertain\". Just give your judgement without any other words.',
        'tail': '\nAnswer: ',
    },
    'qa_emnlp25':{
        'none': 'Answer the following question based on your internal knowledge with one or few words. Provide only one name, with no additional or irrelevant text.\nQuestion: {question}',
        'tail': '\nAnswer: ',
    },
    'llm_judge':{
        'none': 'Judge whether the following answer about the question is correct. If you are sure the answer is correct, say \"certain\". If not, please say \"uncertain\". Just give your judgement without any other words.\nQuestion: {question}.\nAnswer:{prediction}',
        'tail': '\nAnswer: ',
    },
    'llm_judge_long_reason':{
        'none': 'Think step by step and judge whether the following answer about the question is correct. If you are sure the answer is correct, say \"certain\". If not, please say \"uncertain\". Give your judgement with your thinking process.\nQuestion: {question}.\nAnswer:{prediction}',
        'tail': '\nAnswer: ',
    },
    'llm_judge_with_think':{
        'none': 'Judge whether the following answer about the question is correct. The content inside <rationale></rationale> represents the reasoning process of the model, while the content after </rationale> is the answer provided by the model. If you are sure the answer is correct, say \"certain\". If not, please say \"uncertain\". Just give your judgement without any other words.\nQuestion: {question}.\nAnswer:{prediction}',
        'tail': '\nAnswer: ',
    },
    'llm_judge_with_think_long_reason':{
        'none': 'Think step by step and and judge whether the following answer about the question is correct. The content inside <think></think> represents the reasoning process of the model, while the content after </think> is the answer provided by the model. If you are sure the answer is correct, say \"certain\". If not, please say \"uncertain\". Give your judgement with your thinking process.\nQuestion: {question}.\nAnswer:{prediction}',
        'tail': '\nAnswer: ',
    },
    'llm_judge_with_think_reason':{
        'none': 'The content inside <think></think> represents the reasoning process of the model, while the content after </think> is the answer provided by the model. Previously, when no reasoning process was given, you judged that the answer was correct. Now, after being provided with the reasoning process, you have judged that the answer is actually incorrect. Explain clearly why your judgement changed. Provide a step-by-step explanation of how the given reasoning process revealed flaws or inconsistencies that led you to conclude the answer is wrong. Describe which parts of the reasoning you identified as problematic, how these issues connect to the final answer, and why they demonstrate that the answer cannot be considered correct. Be detailed and explicit so that it is clear how the provided reasoning process guided your revised judgement.\nQuestion: {question}.\nAnswer:{prediction}',
        'tail': '\nAnswer: ',
    },
    'llm_judge_with_think_reason_confidence':{
        'none': 'The content inside <think></think> represents the reasoning process of the model, while the content after </think> is the answer provided by the model. Previously, when no reasoning process was given, you judged the answer as uncertain. Now, after being provided with the reasoning process, you have judged that the answer is actually correct (certain). Explain clearly why your judgement changed. Provide a step-by-step explanation of how the given reasoning process offered supporting evidence, clarified ambiguities, or resolved doubts that led you to conclude the answer is correct. Describe which parts of the reasoning you identified as reliable or persuasive, how these strengthened the final answer, and why they demonstrate that the answer can now be considered correct. Be detailed and explicit so that it is clear how the provided reasoning process guided your revised judgement.\nQuestion: {question}.\nAnswer:{prediction}',
        'tail': '\nAnswer: ',
    },
    'llm_judge_and_answer': {
        'none': (
            "You are both a judge and an answerer.\n"
            "Step 1: Based on your knowledge, first give your own short answer to the question (one word/phrase/number if possible). "
            "This is your 'judge_answer'.\n"
            "Step 2: Judge whether the following answer about the question is correct. If you are sure the answer is correct, say \"certain\". If not, please say \"uncertain\". Just give your judgement without any other words.\n "
            "Question: {question}\n"
            "Given Answer: {prediction}\n"
            "Judge Answer: <your own answer>\n"
            "Judgement: <certain/uncertain>"
        ),
        'tail': '\n',
    },
    'llm_judge_consistency': {
        'none': (
            "Compare the following two answers to the same question.\n\n"
            "Given Answer: {given_answer}\n"
            "Judge Answer: {judge_answer}\n\n"
            "If they state the same answer (i.e., mean the same thing), output: certain.\n"
            "If they are different answers, output: uncertain.\n\n"
            "Only output the single word: certain or uncertain."
        ),
        'tail': '\nAnswer: ',
    },
}


def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

def write_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f'write jsonl to: {path}')
    f.close()

def get_prompt(prompt_type, question, prediction="", answer="", given_answer="", judge_answer=""):
    prompt = prompt_dict[prompt_type]['none']

    if 'judge_gold' in prompt_type:
        prompt = prompt.format(question=question, prediction=prediction, answer=answer)

    elif prompt_type == 'llm_judge_and_answer':
        prompt = prompt.format(question=question, prediction=prediction)

    elif prompt_type == 'llm_judge_consistency':
        prompt = prompt.format(given_answer=given_answer, judge_answer=judge_answer)

    elif 'post' in prompt_type or 'judge' in prompt_type:
        prompt = prompt.format(question=question, prediction=prediction)

    else:
        prompt = prompt.format(question=question)

    return prompt

class MyDataset:
    def __init__(self, path, dataset_name) -> None:
        self.dataset_name = dataset_name
        print(f'{self.dataset_name}')
        print(f'data path: {path}')
        if '.jsonl' in path:
            self.data = read_json(path)
        elif '.json' in path:
            self.data = json.loads(open(path).read())
        else:
            raise ValueError('need .jsonl or .json file')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:
        return self.data[index]

    def prepare_qa_data(self):
        """
        将每个数据的原始数据转换为qa可用数据
        """
        qa_data = []
        for item in self.data:
            if 'answer' not in item:
                if 'reference' in item:
                    item['answer'] = item['reference']
                else:
                    item['answer'] = ''
            qa_data.append(item)
            
        return qa_data
            
    def prepare_prompts(self, prompt_type):
        prompt_data = self.prepare_qa_data()
        final_data = []            
        for item in prompt_data:
            if 'judge_gold' in prompt_type:
                # 对每个response都构造判断的prompt
                for tmp_response in item['response']:
                    new_item = item.copy()
                    new_item['instruction'] = get_prompt(prompt_type, item['question'], tmp_response.split('</rationale>')[-1].replace('\n\n', ''), item['answer'])
                    final_data.append(new_item)
            elif prompt_type == 'llm_judge_and_answer':
                
                resp = item['response'][0] if isinstance(item['response'], list) else item['response']
                
                given_answer = resp.split('</think>')[-1].strip()
                given_answer = re.sub(r'^\s*Answer:\s*', '', given_answer, flags=re.IGNORECASE).strip()
               
                item['given_answer'] = given_answer

                item['instruction'] = get_prompt(prompt_type, item['question'], given_answer, '')
                final_data.append(item)
                
            elif prompt_type == 'llm_judge_consistency':
                given_answer = item.get('given_answer', '')
                judge_answer = item.get('judge_answer', '')

                item['instruction'] = get_prompt(
                    prompt_type,
                    question=item['question'],
                    given_answer=given_answer,
                    judge_answer=judge_answer
                )
                final_data.append(item)

            elif 'post' in prompt_type or 'judge' in prompt_type: 
                if 'think' in prompt_type:
                    item['instruction'] = get_prompt(prompt_type, item['question'], item['response'][0], item['answer'])
                else:
                    item['instruction'] = get_prompt(prompt_type, item['question'], item['response'][0].split('</rationale>')[-1].replace('\n\n', ''), item['answer'])
                final_data.append(item)
            else:
                item['instruction'] = get_prompt(prompt_type, item['question'])
                if 'qa' in prompt_type and self.dataset_name == 'strategyqa':
                    item['instruction'] += ' You can only answer yes or no.'
                if 'basketball' in self.dataset_name:
                    item['instruction'] = item['instruction'].replace('one name', 'one city name')
                final_data.append(item)
        return final_data
                
if __name__ == '__main__':
        base_path = '/mnt/bn/motor-nlp-team/users/nishiyu/data/'
        data_path = [
            '2wikimultihopqa/dev.json',
            'hq/hotpotqa-dev.json',
            'nq/nq_test.jsonl',
            'strategyqa/strategyqa_train.json',
            'gsm8k/test.jsonl'
        ]
        for name in data_path:
            file_path = base_path + name
            prompt_type = 'qa'
            data = MyDataset(file_path)
            qa_data = data.prepare_qa_data()
            print(f'dataset: {name.split("/")[0]}')
            print(f'data size: {len(data)}')
            for idx in range(5):
                print(f'data example <question, answer>. Q: {qa_data[idx]["question"]}; A: {qa_data[idx]["answer"]}')
            
