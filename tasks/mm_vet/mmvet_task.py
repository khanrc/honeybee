from prettytable import PrettyTable
from tasks.base_task import Task, TaskScore
from .mmvet_eval import gpt_eval


class MMVetScore(TaskScore):
    def get_summary(self, max_level=1):
        # do not support max_level > 1
        ret = {
            "total": self.scores["cap_score"]["total"]
        }
        return ret

    def dumps(self):
        filed_names = [
            "rec", "ocr", "know", "gen", "spat", "math", "total"
        ]

        tb = PrettyTable()
        tb.field_names = filed_names
        tb.align["Type"] = "l"

        tb.add_row([
            self.scores["cap_score"][name] for name in filed_names
        ])

        return tb.get_string()


class MMVetTask(Task):
    def compute_score(self, results):
        root = self.data_loader.dataset.root  # somewhat hacky ...
        results_to_compute = {
            f"v1_{k}": v["pred"] for k, v in results.items()
        }
        summary, grade_results = gpt_eval(root, results_to_compute)

        score = MMVetScore(summary)

        return score
