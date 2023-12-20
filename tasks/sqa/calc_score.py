"""Modified from the official score calculation code:
    https://github.com/lupantech/ScienceQA/blob/main/tools/evaluate_acc.py
"""
import json
import os
import random
import pandas as pd

_OPTIONS = ["A", "B", "C", "D", "E"]


class SQAMetric:
    def __init__(
        self,
        split,
        annotation_base_dir,
    ):
        with open(os.path.join(annotation_base_dir, "problems.json")) as f:
            self.problems = json.load(f)

    def get_acc_with_condition(self, res_pd, key, values):
        if isinstance(values, list):
            total_pd = res_pd[res_pd[key].isin(values)]
        else:
            total_pd = res_pd[res_pd[key] == values]
        correct_pd = total_pd[total_pd['true_false'] == True]
        if len(total_pd) == 0:
            return -1
        acc = len(correct_pd) / len(total_pd) * 100
        return acc

    def get_pred_idx(self, prediction, choices, options):
        """
        Get the index (e.g. 2) from the prediction (e.g. 'C') with additional random guess.
        From llava: https://github.com/haotian-liu/LLaVA/blob/82fc5e0e5f4393a4c26851fa32c69ab37ea3b146/llava/eval/eval_science_qa.py#L28 
        """
        if prediction in options[: len(choices)]:
            return options.index(prediction)
        else:
            return random.choice(range(len(choices)))

    def process_result(self, results_dic):
        # read result file
        results_by_qid = {}
        for item in results_dic.values():
            results_by_qid[item['question_id']] = item
        num = len(results_by_qid)
        assert num == 4241

        # construct pandas for meta data
        sqa_pd = pd.DataFrame(self.problems).T
        res_pd = sqa_pd[sqa_pd['split'] == 'test']  # test set

        # update data
        for index, row in res_pd.iterrows():

            res_pd.loc[index, 'no_context'] = True if (not row['hint'] and not row['image']) else False
            res_pd.loc[index, 'has_text'] = True if row['hint'] else False
            res_pd.loc[index, 'has_image'] = True if row['image'] else False
            res_pd.loc[index, 'has_text_image'] = True if (row['hint'] and row['image']) else False

            label = row['answer']
            pred = self.get_pred_idx(results_by_qid[index]['pred'], row.choices, _OPTIONS)
            res_pd.loc[index, 'pred'] = pred
            res_pd.loc[index, 'true_false'] = (label == pred)

        # accuracy scores
        acc_average = len(res_pd[res_pd['true_false'] == True]) * 100 / num

        scores = {
            'acc_natural':
            self.get_acc_with_condition(res_pd, 'subject', 'natural science'),
            'acc_social':
            self.get_acc_with_condition(res_pd, 'subject', 'social science'),
            'acc_language':
            self.get_acc_with_condition(res_pd, 'subject', 'language science'),
            'acc_has_text':
            self.get_acc_with_condition(res_pd, 'has_text', True),
            'acc_has_image':
            self.get_acc_with_condition(res_pd, 'has_image', True),
            'acc_no_context':
            self.get_acc_with_condition(res_pd, 'no_context', True),
            'acc_grade_1_6':
            self.get_acc_with_condition(res_pd, 'grade', ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']),
            'acc_grade_7_12':
            self.get_acc_with_condition(res_pd, 'grade', ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']),
            'acc_average': acc_average,
        }
        topics = ['punctuation', 'literacy-in-science', 'verbs', 'pronouns', 'civics', 'culture', 'word-study', 'economics', 'physics', 'units-and-measurement', 'science-and-engineering-practices', 'reading-comprehension', 'global-studies', 'grammar', 'figurative-language', 'us-history', 'writing-strategies', 'world-history', 'reference-skills', 'biology', 'earth-science', 'phonological-awareness', 'capitalization', 'chemistry', 'vocabulary', 'geography']
        for t in topics:
            scores['acc_' + t] = self.get_acc_with_condition(res_pd, 'topic', t)

        return scores
