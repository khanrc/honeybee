from prettytable import PrettyTable
from tasks.base_task import Task, TaskScore
from .gpt_eval import gpt_eval


class LlavaBenchScore(TaskScore):
    def get_summary(self, max_level=1):
        ret = {
            k: scores["score"]
            for k, scores in self.scores.items()
        }
        return ret

    def dumps(self):
        tb = PrettyTable()
        tb.field_names = ["Type", "All", "Complex", "Conv", "Detail"]
        tb.align["Type"] = "l"

        for mtype in self.scores["all"].keys():
            tb.add_row(
                [
                    mtype,
                    self.scores["all"][mtype],
                    self.scores["complex"][mtype],
                    self.scores["conv"][mtype],
                    self.scores["detail"][mtype],
                ],
                divider=(mtype=="all"),
            )

        return tb.get_string()


class LlavaBenchTask(Task):
    def compute_score(self, results):
        root = self.data_loader.dataset.root
        summary = gpt_eval(root, results)

        score = LlavaBenchScore(summary)

        return score
