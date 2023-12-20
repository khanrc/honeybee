import os

import pandas as pd
import prettytable

from tasks.base_task import Task, TaskScore
from tasks.mmb.calc_score import MMBMetric


class MMBScore(TaskScore):
    def get_summary(self, max_level=1):
        ret = {
            "total_acc": float(self.scores["total_acc"]),
        }
        if max_level >= 2:
            for k, v in self.scores.items():
                if not isinstance(v, dict):
                    continue

                ret[f"{k}"] = float(v["parent_task_acc"])

                if max_level >= 3:
                    ret.update(
                        {
                            f"{k}/{k2}": float(v2["acc"])
                            for k2, v2 in v.items()
                            if isinstance(v2, dict)
                        }
                    )

        return ret

    def dumps(self):
        def item_painting(item, color):
            color_code = {
                "black": "\u001b[30m",
                "red": "\u001b[31m",
                "green": "\u001b[32m",
                "yellow": "\u001b[33m",
                "blue": "\u001b[34m",
                "magenta": "\u001b[35m",
                "cyan": "\u001b[36m",
                "white": "\u001b[37m",
                "reset": "\u001b[0m",
            }
            return f"{color_code[color.lower()]}{item}\u001b[0m"

        tb = prettytable.PrettyTable()
        tb.field_names = ["Task", "Num samples", "Num corrects", "Acc"]
        tb.float_format = ".2"
        for eval_type, dic in self.scores.items():
            if isinstance(dic, dict):
                # Parent category
                acc = dic["parent_task_acc"]
                n_samples = dic["parent_num_samples"]
                n_corrects = dic["parent_num_correct"]

                parent_row = [eval_type, n_samples, n_corrects, acc]
                parent_row = [item_painting(elem, "yellow") for elem in parent_row]
                tb.add_row(parent_row, divider=True)

                # specific results for each leaf category
                for i, (k, v) in enumerate(dic.items()):
                    if isinstance(v, dict):
                        tb.add_row(
                            [k, v["num_sampes"], v["num_correct"], v["acc"]],
                            divider=(i == len(dic) - 4),
                        )

        total_row = [
            "Total",
            self.scores["total_samples"],
            self.scores["total_corrects"],
            self.scores["total_acc"],
        ]
        total_row = [item_painting(elem, "green") for elem in total_row]
        tb.add_row(total_row)

        return tb.get_string()


class MMBTask(Task):
    answer_prefix = "The answer is "

    def dump_submission_file(self, result_dir, results):
        parsed_data = []
        for key in results:
            item = results[key]
            options = [v if v != "N/A" else "" for k, v in item["options_dict"].items()]
            pred = item["pred"]
            if self.answer_prefix in pred:
                pred = pred[len(self.answer_prefix) :].replace(".", "")

            item_row = [
                item["index"],
                item["question"],
                pred,
                item["category"],
                item["l2_category"],
                item["answer"],
            ]
            item_row.extend(options[:-1])
            parsed_data.append(item_row)

        df = pd.DataFrame(
            parsed_data,
            columns=[
                "index",
                "question",
                "prediction",
                "category",
                "l2_category",
                "answer",
                "A",
                "B",
                "C",
                "D",
            ],
        )
        df.to_excel(os.path.join(result_dir, "MMB_submission.xlsx"), index=False, engine="openpyxl")

    def compute_score(self, results):
        # compute metric
        metric = MMBMetric()
        res = metric.process_result(results)
        score = MMBScore(res)

        return score
