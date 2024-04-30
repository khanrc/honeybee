from collections import defaultdict
import numpy as np

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

    for k, v in sorted(scores.items()):
        stats = np.asarray(v).mean(0).tolist()
        stats = [round(x, 3) for x in stats]
        print(k, round(stats[1]/stats[0]*100, 1), round(stats[0] * 10, 1), round(stats[1] * 10, 1))
    print('=================================')
