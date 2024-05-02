import os

import prettytable

from tasks.base_task import Task, TaskScore
from tasks.sqa.calc_score import SQAMetric


class SQAScore(TaskScore):
    def get_summary(self, max_level=1):
        return {"acc": self.scores["acc"]}

    def dumps(self):
        tb = prettytable.PrettyTable()
        tb.field_names = list(self.scores.keys())
        tb.float_format = ".2"
        total_scores = []
        for _, value in self.scores.items():
            total_scores.append(value)
        tb.add_row(total_scores, divider=True)

        return tb.get_string()


class SQATask(Task):
    def compute_score(self, results):
        # compute metric
        metric = SQAMetric(
            split=self.data_loader.dataset.split,
            annotation_base_dir=os.path.join(self.data_loader.dataset.root, "text"),
        )
        res = metric.process_result(results)
        score = SQAScore(res)

        return score
