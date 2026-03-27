import json
from utils import has_answer


def deal_judge_new(pred):
    
    if pred is None:
        return True   
    if has_answer([
        "sorry", "apologize", "apologies", "uncertain", "false", "no", "unsure",
        "cannot", "unknown", "no specific answer", "not provide", "cannot answer",
        "no information provided", "no answer", "not contain", "no definitive answer"
    ], pred):
        return True   
    return False      

def compute_overall_accuracy_with_dealer(file_path):

    total = 0
    certain_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            pred = data['response'][0].strip().lower()
            is_uncertain = deal_judge_new(pred)
            
            if not is_uncertain:
                certain_count += 1
            total += 1

    accuracy = certain_count / total if total else 0.0
    return certain_count, total, accuracy

if __name__ == '__main__':
    file_path=""
    certain, total, acc = compute_overall_accuracy_with_dealer(file_path)
    print(f"样本总数: {total}")
    print(f"certain 的数量: {certain}")
    print(f"总体正确率 (accuracy): {acc:.2%}")
