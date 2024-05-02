"""https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/eval_gpt_review_bench.py
"""
from pathlib import Path
from collections import defaultdict
import time

import numpy as np
import openai

import utils

NUM_SECONDS_TO_SLEEP = 10

# Please fill in the following information to use the OpenAI API
openai.api_type = "azure"
openai.api_base = "" # Fill you api url
openai.api_version = "2023-07-01-preview"
openai.api_key = ""  # Fill your api key 
gpt_engine = ""  # Fill your engine name for gpt4


def get_eval(content: str, max_tokens: int = 20):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4-0314',
                engine=gpt_engine,
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,
                max_tokens=max_tokens,
                stop=["\n"],
            )
            break
        except Exception as e:
            print(f"[ERROR from openai API] {e}")
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response['choices'][0]['message']['content']


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


def summarize(reviews):
    """https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/summarize_gpt_review.py
    """
    scores = defaultdict(list)
    for review in reviews:
        if 'category' in review:
            scores[review['category']].append(review['tuple'])
            scores['all'].append(review['tuple'])
        else:
            if 'tuple' in review:
                scores['all'].append(review['tuple'])
            else:
                scores['all'].append(review['score'])

    results = {}
    for k, v in sorted(scores.items()):
        stats = np.asarray(v).mean(0).tolist()
        stats = [round(x, 3) for x in stats]
        score, gpt_score, pred_score = (
            round(stats[1]/stats[0]*100, 1),
            round(stats[0] * 10, 1),
            round(stats[1] * 10, 1)
        )

        res_key = k.replace("llava_bench_", "")
        results[res_key] = {
            "score": score,
            "gpt": gpt_score,
            "pred": pred_score,
        }
        print(k, score, gpt_score, pred_score)

    return results


def gpt_eval(root: str | Path, results: dict):
    root = Path(root)
    question_path = root / "questions.jsonl"
    context_path = root / "context.jsonl"
    rule_path = root / "rule.json"
    gpt_answer_path = root / "answers_gpt4.jsonl"

    questions = utils.load(question_path)
    answers1 = utils.load(gpt_answer_path)
    answers2 = [
        {"question_id": dic["question_id"], "text": dic["pred"], "category": dic["category"]}
        for dic in results.values()
    ]

    rule_dict = utils.load(rule_path)

    context_list = utils.load(context_path)
    image_to_context = {context['image']: context for context in context_list}

    reviews = []
    idx = 0
    assert len(questions) == len(answers1) == len(answers2)
    N = len(questions)
    for ques, ans1, ans2 in zip(questions, answers1, answers2):
        inst = image_to_context[ques['image']]

        if isinstance(inst['caption'], list):
            cap_str = '\n'.join(inst['caption'])
        else:
            cap_str = inst['caption']

        category = 'llava_bench_' + ques['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            assert False, f"Visual QA category not found in rule file: {category}."
        prompt = rule['prompt']
        role = rule['role']
        content = (f'[Context]\n{cap_str}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        cur_js = {
            'id': idx+1,
            'question_id': ques['question_id'],
            'category': category
        }
        review = get_eval(content)
        scores = parse_score(review)
        cur_js['content'] = review
        cur_js['tuple'] = scores
        reviews.append(cur_js)

        idx += 1
        print(f"[{idx}/{N}] Review: {review} | Parsed scores: {scores}")

    summary = summarize(reviews)
    return summary
