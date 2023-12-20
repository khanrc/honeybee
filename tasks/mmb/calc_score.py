"""Modified from the evaluation code:
    https://github.com/open-compass/opencompass/blob/main/tools/eval_mmbench.py
"""
import random
import pandas as pd
import string

_OPTIONS = ["A", "B", "C", "D"]

# Utils
def double_log(msg, fout=None):
    print(msg)
    if fout is not None:
        fout.write(str(msg) + "\n")
        fout.flush()

def build_choices(item):
    ret = {}
    for ch in "ABCD":
        if not pd.isna(item[ch]):
            ret[ch] = item[ch]
    return ret


# Prefetch Answers
def can_infer_option(answer, num_choice=5):
    choices = string.ascii_uppercase[:num_choice]
    if "Failed to obtain answer via API" in answer:
        return False

    def count(splits, choices="ABCD", prefix="", suffix=""):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    splits = [x.strip() for x in answer.split()]
    if count(splits, choices) == 1:
        for ch in choices:
            if "A" in splits and len(splits) > 3:
                double_log(f"A might be a quantifier in the string: {answer}. ", fout)
                break
            if ch in splits:
                return ch
    tups = [
        ("", "."),
        ("", ","),
        ("", ":"),
        ("", ")"),
        ("", ")."),
        ("(", ")"),
        ("(", ")."),
        (":", ""),
        (":", ","),
        (":", "."),
        (":", ")"),
        (":", ")."),
    ]
    for tup in tups:
        if count(splits, choices, prefix=tup[0], suffix=tup[1]) == 1:
            for ch in choices:
                if tup[0] + ch + tup[1] in splits:
                    return ch
    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in "ABCD"
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    copt = can_infer_option(answer)
    return copt if copt else can_infer_text(answer, choices)


def prefetch_answer(item):
    choices = build_choices(item)
    return can_infer(item["prediction"], choices)


class MMBMetric:
    answer_prefix = "The answer is "

    def parse_answer(self, pred_text, choices, options):
        answer = "FAILED"
        for i, cand in enumerate(choices):
            if cand in pred_text:
                answer = options[i]

        if answer == "FAILED":
            answer = random.choice(_OPTIONS)
        return answer  # return {'A', ..., 'E', or FAILED}

    def compute_metric(self, scores_dict):
        total_samples = 0
        total_corrects = 0
        for parent_key in scores_dict:
            parent_task_samples = 0
            parent_task_corrects = 0
            for child_key in scores_dict[parent_key]:
                num_samples = scores_dict[parent_key][child_key]["num_sampes"]
                num_correct = scores_dict[parent_key][child_key]["num_correct"]
                scores_dict[parent_key][child_key]["acc"] = f"{num_correct / num_samples * 100:.2f}"
                parent_task_samples += num_samples
                parent_task_corrects += num_correct
            scores_dict[parent_key][
                "parent_task_acc"
            ] = f"{parent_task_corrects / parent_task_samples * 100:.2f}"
            scores_dict[parent_key]["parent_num_samples"] = parent_task_samples
            scores_dict[parent_key]["parent_num_correct"] = parent_task_corrects
            total_samples += parent_task_samples
            total_corrects += parent_task_corrects

        scores_dict["total_samples"] = total_samples
        scores_dict["total_corrects"] = total_corrects
        scores_dict["total_acc"] = f"{total_corrects / total_samples * 100:.2f}"
        return scores_dict

    def circular_eval(self, circular_samples):
        GT, PRED = [], []

        for i in range(len(circular_samples)):
            item = circular_samples[i]
            GT.append(item["answer"])

            item_dict = {
                "index": item["index"],
                "question": item["question"],
                "category": item["category"],
                "l2_category": item["l2_category"],
                "answer": item["answer"],
            }
            for k, v in item["options_dict"].items():
                item_dict[k] = v if v != "N/A" else ""

            pred = item["pred"]
            if self.answer_prefix in pred:
                pred = pred[len(self.answer_prefix) :].replace(".", "")
            item_dict["prediction"] = pred
            PRED.append(prefetch_answer(item_dict))

            if PRED[-1] and (GT[-1] != PRED[-1]):
                return 0

        for i in range(len(circular_samples)):
            if PRED[i]:
                continue
            else:
                ret = random.choice(GT)
            PRED[i] = ret
            if PRED[i] != GT[i]:
                return 0

        return 1

    def process_result(self, results_dic, verbose=False):
        random.seed(2680)
        scores = {}  # score dictionary for printing scores

        # Reconstruct result dict to be compatible with circular eval.
        predictions = {}
        for pred in results_dic.values():
            # Indices of circular samples for a given image are like 57, 100057, 200057, ...
            main_index = pred["index"] % int(1e6)
            if main_index not in predictions:
                predictions[main_index] = [pred]
            else:
                predictions[main_index].append(pred)

        for (
            _,
            example,
        ) in predictions.items():  # Each example contanins multiple circular evaluation samples.
            is_hit = self.circular_eval(example)

            category_child = example[0]["category"]
            category_parent = example[0]["l2_category"]

            if is_hit:  # for the case of correct answer
                if category_parent in scores:
                    if category_child in scores[category_parent]:
                        scores[category_parent][category_child]["num_sampes"] += 1
                        scores[category_parent][category_child]["num_correct"] += 1
                    else:
                        temp_score_dict = {"num_sampes": 1, "num_correct": 1}
                        scores[category_parent][category_child] = temp_score_dict
                else:
                    temp_score_dict = {"num_sampes": 1, "num_correct": 1}
                    scores[category_parent] = {category_child: temp_score_dict}
            else:  # for the case of incorrect answer
                if category_parent in scores:
                    if category_child in scores[category_parent]:
                        scores[category_parent][category_child]["num_sampes"] += 1
                    else:
                        temp_score_dict = {"num_sampes": 1, "num_correct": 0}
                        scores[category_parent][category_child] = temp_score_dict
                else:
                    temp_score_dict = {"num_sampes": 1, "num_correct": 0}
                    scores[category_parent] = {category_child: temp_score_dict}

        scores = self.compute_metric(scores)
        print(
            f'Total: {scores["total_samples"]}, Correct: {scores["total_corrects"]}, Accuracy: {scores["total_acc"]}%'
        )

        return scores
