import os

import prettytable

from tasks.base_task import Task, TaskScore
from tasks.sqa.calc_score import SQAMetric


class SQAScore(TaskScore):
    summary_targets = [
        'acc_natural', 'acc_social', 'acc_language',
        'acc_has_text', 'acc_has_image', 'acc_no_context',
        'acc_grade_1_6', 'acc_grade_7_12', 'acc_average',
    ]

    def get_summary(self, max_level=1):
        summary_dict = {}
        for k, v in self.scores.items():
            if k in self.summary_targets:
                summary_dict[k] = v
        return summary_dict

    def dumps(self):
        tb = prettytable.PrettyTable()
        tb.field_names = ["SQA Tasks", "Acc"]
        tb.float_format = ".2"
        total_scores = []
        for key, value in self.scores.items():
            is_divider = True if key == "acc_average" else False
            tb.add_row([key, value], divider=is_divider)

        tb.align["Tasks"] = "l"
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
