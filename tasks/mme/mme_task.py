from collections import defaultdict
from prettytable import PrettyTable

from tasks.base_task import Task, TaskScore
from tasks.mme.calc_score import MMEMetric


class MMEScore(TaskScore):
    def get_summary(self, max_level=2):
        ret = {
            "total": self.scores["Total"][0],
        }
        if max_level >= 2:
            ret.update(
                {
                    "perception": self.scores["Perception"]["Total"][0],
                    "cognition": self.scores["Cognition"]["Total"][0],
                }
            )
        if max_level >= 3:
            ret.update(
                {
                    f"perception/{k}": v[0]
                    for k, v in self.scores["Perception"].items()
                    if k != "Total"
                }
            )
            ret.update(
                {
                    f"cognition/{k}": v[0]
                    for k, v in self.scores["Cognition"].items()
                    if k != "Total"
                }
            )

        return ret

    def dumps(self):
        tb = PrettyTable()
        tb.field_names = ["MME", "Score", "Acc", "Acc+"]
        tb.align["MME"] = "l"
        tb.float_format = ".2"
        for eval_type, dic in self.scores.items():
            if not isinstance(dic, dict):
                continue

            # Perception / Cognition
            score, acc, acc_plus = dic["Total"]
            tb.add_row([eval_type, score, acc, acc_plus], divider=True)
            # specific results for each task
            for i, (k, v) in enumerate(dic.items()):
                if k == "Total":
                    continue
                tb.add_row([f"{eval_type}/{k}", *v], divider=(i == len(dic) - 2))

        score, acc, acc_plus = self.scores["Total"]
        tb.add_row(["Total", score, acc, acc_plus])

        return tb.get_string()


class MMETask(Task):
    def compute_score(self, results):
        # 1. integrate outputs
        outputs = defaultdict(list)

        for m in results.values():
            question = m["question"]
            answer = m["answer"]
            image_id = m["image_id"]
            dataset_name = m["dataset_name"]
            pred = m["pred"]

            outputs[dataset_name].append("\t".join((image_id, question, answer, pred)))

        # 2. compute metric
        metric = MMEMetric()
        res = metric.process_result(outputs)
        score = MMEScore(res)

        return score
