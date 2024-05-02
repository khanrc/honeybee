from prettytable import PrettyTable
from tasks.base_task import Task, TaskScore
from .calc_score import eval_pope


class POPEScore(TaskScore):
    def get_summary(self, max_level=1):
        ret = {}
        for category, subscores in self.scores.items():
            for key, value in subscores.items():
                if key in ["TP", "TN", "FP", "FN"]:
                    continue

                ret[f"{category}/{key}"] = value

        return ret

    def dumps(self):
        tb = PrettyTable()
        keys = list(self.scores["adversarial"].keys())
        tb.field_names = ["Category", *keys]
        tb.float_format = ".3"
        avg = {}
        n_categories = len(self.scores)
        for i, (category, subscores) in enumerate(self.scores.items()):
            is_last = i == (len(self.scores)-1)
            tb.add_row([category, *list(subscores.values())], divider=is_last)
            # compute average
            for key, subscore in subscores.items():
                avg.setdefault(key, 0)
                avg[key] += subscore / n_categories

        for key, score in avg.items():
            if key in ["TP", "TN", "FP", "FN"]:
                avg[key] = int(score)

        tb.add_row(["average", *list(avg.values())])

        return tb.get_string()


class POPETask(Task):
    def compute_score(self, results: dict) -> POPEScore:
        categories = [
            "adversarial",
            "popular",
            "random",
        ]
        cat_results = {category: [] for category in categories}
        for res in results.values():
            cat_results[res["category"]].append(res)

        scores = {category: {} for category in categories}
        for category, results in cat_results.items():
            label_list = [res["answer"] for res in results]
            preds = [{"text": res["pred"]} for res in results]
            res = eval_pope(preds, label_list)
            scores[category] = res

        scores = POPEScore(scores)
        return scores
