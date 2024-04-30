import random

from tasks.mmb.eval_mmb_gpt import prefetch_answer

_OPTIONS = ["A", "B", "C", "D"]


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
