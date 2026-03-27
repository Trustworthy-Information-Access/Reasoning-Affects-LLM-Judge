import json
import random

prompt_dict = {
    'qa_pop_rank_diverse': {
        'none': "Rate how familiar you are with the {entity_type} '{entity_name}'. The familiarity is rated on a scale from 1 to 10, where 10 means you are highly familiar with it, and 1 means you have little to no knowledge about it. Your answer needs to be a precise integer. Provide only the number, without any additional explanation.",
        'tail': '\nNumber: ',
    },
    'qa_coo_rank_diverse': {
        'none': "Rate how familiar you are with the relationship between the {question_entity_type} '{question_entity_name}' and the {gene_entity_type} '{gene_entity_name}'. The familiarity is rated on a scale from 1 to 10, where 10 means you are highly familiar with their relationship, and 1 means you know little to nothing about it. Your answer needs to be a precise integer. Provide only the number, without any additional explanation.",
        'tail': '\nNumber: ',
    }
}


def get_prompt(sample, args, few_shot_examples):
    "question_pop, gene_pop, coo_pop"
    dataset_need_names = {
        'movies': {'question': 'movie', 'gene': 'director', 'coo': ['movie', 'director']},
        'songs': {'question': 'song', 'gene': 'performer', 'coo': ['song', 'performer']},
        'basketball': {'question': 'basketball palyer', 'gene': 'city', 'coo': ['basketball player', 'city']},
    }
    prompt = prompt_dict[args.type]['none'] # prior
    tail = prompt_dict[args.type]['tail']
    need_names = dataset_need_names[args.dataset_name][args.gene_type]

    if args.gene_type in ['question', 'gene']:
        base_prompt = prompt.format(entity_type=need_names, entity_name=sample[args.gene_type + '_entity'])
    elif args.gene_type == 'coo':
        base_prompt = prompt.format(question_entity_type=need_names[0], question_entity_name=sample['question_entity'], gene_entity_type=need_names[1], gene_entity_name=sample['gene_entity'])


    few_shot_prompt = ""
    if len(few_shot_examples) > 0:
        few_shot_prompt = "\nHere are some examples:\n"
        for item in few_shot_examples:
            if args.gene_type == 'question':
                few_shot_prompt += f'The {need_names}: {item["question_entity"]}\n'
                few_shot_prompt += f'Number: {item["question_pop_level"]}\n'
            elif args.gene_type == 'gene':
                few_shot_prompt += f'The {need_names}: {item["gene_entity"]}\n'
                few_shot_prompt += f'Number: {item["gene_pop_level"]}\n'
            elif args.gene_type == 'coo':
                few_shot_prompt += f'The {need_names[0]}: {item["question_entity"]}; The {need_names[1]}: {item["gene_entity"]}\n'
                few_shot_prompt += f'Number: {item["coo_pop_level"]}\n'
        few_shot_prompt += base_prompt
    final_prompt = base_prompt + few_shot_prompt + tail
        
    # 每个模型特有的prompt格式
    return final_prompt






