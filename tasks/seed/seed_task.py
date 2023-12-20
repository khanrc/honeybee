from prettytable import PrettyTable
from tasks.base_task import Task, TaskScore


class SEEDScore(TaskScore):
    def get_summary(self, max_level=1):
        ret = {
            "total": self.scores["Total"]["acc"],
        }
        if max_level >= 2:
            for q_type, scores in self.scores.items():
                if q_type == "Total":
                    continue
                q_type_tb = q_type.lower().replace(" ", "_")
                ret[q_type_tb] = scores["acc"]

        return ret

    def dumps(self):
        tb = PrettyTable(["Type", "Acc", "Correct", "Total"])
        tb.align["Type"] = "l"
        for i, (q_type, scores) in enumerate(self.scores.items()):
            acc = scores["acc"]
            acc = f"{acc:.1f}"
            divider = i == len(self.scores) - 2  # add divider above total
            tb.add_row([q_type, acc, scores["correct"], scores["total"]], divider=divider)

        return tb.get_string()


def calc_seed_score(results: dict):
    # Modified from official code
    # https://github.com/AILab-CVC/SEED-Bench/blob/main/eval.py#L125-L147
    def preproc(s):
        return s.strip().lower()

    type_counts = {}
    correct_counts = {}
    for item in results.values():
        pred = preproc(item["pred"])
        gt = preproc(item["answer"])
        data_type = item["question_type"]

        type_counts[data_type] = type_counts.get(data_type, 0) + 1
        correct_counts[data_type] = correct_counts.get(data_type, 0) + int(pred == gt)

    total_count = 0
    total_correct = 0
    for data_type in type_counts.keys():
        total_count += type_counts[data_type]
        total_correct += correct_counts[data_type]

    # collect q_types, sorted by q_id, from results
    q_types = {
        item["question_type_id"]: item["question_type"]
        for item in results.values()
    }
    q_types = [q_types[q_id] for q_id in sorted(q_types.keys())]

    scores = {
        q_type: {
            "total": type_counts[q_type],
            "correct": correct_counts[q_type],
            "acc": correct_counts[q_type] / type_counts[q_type] * 100,
        }
        for q_type in q_types
    }
    scores["Total"] = {
        "total": total_count,
        "correct": total_correct,
        "acc": total_correct / total_count * 100,
    }

    return scores


class SEEDTask(Task):
    def compute_score(self, results: dict) -> SEEDScore:
        scores = calc_seed_score(results)
        scores = SEEDScore(scores)

        return scores
