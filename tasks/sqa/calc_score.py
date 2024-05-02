import json
import os
import random

_OPTIONS = ["A", "B", "C", "D", "E"]


class SQAMetric:
    def __init__(
        self,
        split,
        annotation_base_dir,
    ):
        with open(os.path.join(annotation_base_dir, "pid_splits.json")) as f:
            self.split_indices = json.load(f)[split]

        with open(os.path.join(annotation_base_dir, "problems.json")) as f:
            self.problems = json.load(f)

    def get_pred_idx(self, prediction, choices, options):
        """
        Get the index (e.g. 2) from the prediction (e.g. 'C')
        """
        if prediction in options[: len(choices)]:
            return options.index(prediction)
        else:
            return random.choice(range(len(choices)))

    def parse_answer(self, pred_text, choices, options):
        answer = "FAILED"
        for i, cand in enumerate(choices):
            if cand in pred_text:
                answer = options[i]
        return answer  # return {'A', ..., 'E', or FAILED}

    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        return 0

    def process_result(self, results_dic):
        results = {"correct": [], "incorrect": [], "text": [], "image": []}
        sqa_results = {}  # noqa: SIM904
        sqa_results["acc"] = None
        sqa_results["correct"] = None
        sqa_results["count"] = None
        sqa_results["results"] = {}
        sqa_results["outputs"] = {}

        predictions = {pred["question_id"]: pred for pred in results_dic.values()}
        split_problems = {idx: self.problems[idx] for idx in self.split_indices}
        for prob_id, prob in split_problems.items():
            if prob_id not in predictions:
                continue
            pred = predictions[prob_id]
            pred_text = pred["pred"]
            # parse pred_test to the target answer style
            mapping = [("(", ""), (")", ""), (".", "")]
            for k, v in mapping:
                pred_text = pred_text.replace(k, v)

            if len(pred_text) == 1:
                answer = pred_text[0]  # 'A', 'B', ...
            else:
                answer = self.parse_answer(pred_text, prob["choices"], _OPTIONS)

            pred_idx = self.get_pred_idx(answer, prob["choices"], _OPTIONS)

            analysis = {
                "question_id": prob_id,
                "parsed_ans": answer,
                "ground_truth": _OPTIONS[prob["answer"]],
                "question": pred["prompt"],
                "pred": pred_text,
                "is_multimodal": "<image>" in pred["prompt"],
            }

            sqa_results["results"][prob_id] = self.get_pred_idx(answer, prob["choices"], _OPTIONS)
            sqa_results["outputs"][prob_id] = pred_text

            is_image = not (pred["image_path"] is None or pred["image_path"] == "None")
            correct = pred_idx == prob["answer"]
            if is_image:
                results["image"].append(correct)
            else:
                results["text"].append(correct)

            if correct:
                results["correct"].append(analysis)
            else:
                results["incorrect"].append(analysis)

        correct = len(results["correct"])
        total = len(results["correct"]) + len(results["incorrect"])
        print(f"Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%")

        sqa_results["acc"] = correct / total * 100
        sqa_results["correct"] = correct
        sqa_results["count"] = total

        # image/text
        n_image = len(results["image"])
        image_correct = sum(results["image"])
        image_acc = image_correct / n_image * 100 if n_image > 0 else 0
        print(f"[Image] Total: {n_image}, Correct: {image_correct}, Accuracy: {image_acc:.2f}%")
        n_text = len(results["text"])
        text_correct = sum(results["text"])
        text_acc = text_correct / n_text * 100 if n_text > 0 else 0
        print(f"[Text] Total: {n_text}, Correct: {text_correct}, Accuracy: {text_acc:.2f}%")
        assert n_image + n_text == total

        result_acc = {
            "acc": sqa_results["acc"],
            "correct": sqa_results["correct"],
            "incorrect": len(results["incorrect"]),
            "count": sqa_results["count"],
            "acc-image": image_acc,
            "acc-text": text_acc,
        }

        return result_acc
