import random
from prettytable import PrettyTable
import numpy as np

from tasks.base_task import Task, TaskScore
from .mmmu_utils.data_utils import CAT_SHORT2LONG, DOMAIN_CAT2SUB_CAT
from .mmmu_utils.eval_utils import evaluate, parse_open_response, calculate_ins_level_acc



class MMMUScore(TaskScore):
    def get_summary(self, max_level=1):
        ret = {}
        return ret

    def dumps(self):
        tb = PrettyTable(['Subject', 'Data Num', 'Acc'])
        # add domain Subject
        for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
            in_domain_cat_results = {}
            for cat_name in in_domain_cats: # use the order in DOMAIN_CAT2SUB_CAT
                if cat_name in self.scores.keys():
                    in_domain_cat_results[cat_name] = self.scores[cat_name]
                else:
                    pass
            in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
            in_domain_data_num = np.sum([cat_results['num_example'] for cat_results in in_domain_cat_results.values()])

            tb.add_row(['Overall-' + domain, int(in_domain_data_num), round(in_domain_ins_acc, 3)])
            # add sub category
            for cat_name, cat_results in in_domain_cat_results.items():
                tb.add_row([cat_name, int(cat_results['num_example']), round(cat_results['acc'], 3)])

        all_ins_acc = calculate_ins_level_acc(self.scores)
        tb.add_row([
            'Overall',
            np.sum([cat_results['num_example'] for cat_results in self.scores.values()]),
            round(all_ins_acc, 3)
        ])

        return tb.get_string()


class MMMUTask(Task):
    def compute_score(self, results: dict) -> MMMUScore:
        # https://github.com/MMMU-Benchmark/MMMU/blob/main/eval/main_eval_only.py
        output_dict_w_cat = {}
        answer_dict_w_cat = {}

        for res in results.values():
            data_id = res["id"]
            parsed_pred = res["pred"]

            if res["question_type"] == "multiple-choice":
                # for MC, add random guess logic
                multi_images_in_option = res["num_option_images"]
                out_of_choice = not parsed_pred in res["all_choices"]
                if multi_images_in_option or out_of_choice:
                    # random guess for multi-image case
                    parsed_pred = random.choice(res["all_choices"])
                    print(
                        f" >> Random guess (is_multi={multi_images_in_option}, ooc={out_of_choice}): "
                        f"#image={res['num_option_images']}, pred={res['pred']} "
                        f"guess={parsed_pred}, gt={res['answer']}, q_type={res['question_type']}"
                    )

            category = "_".join(data_id.split("_")[1:-1])

            if category not in output_dict_w_cat:
                output_dict_w_cat[category] = {}
            if category not in answer_dict_w_cat:
                answer_dict_w_cat[category] = {}

            output_dict_w_cat[category][data_id] = parsed_pred
            answer_dict_w_cat[category][data_id] = {
                "question_type": res["question_type"],
                "ground_truth": res["answer"],
            }

        evaluation_result = {}

        for category in CAT_SHORT2LONG.values():
            print("Evaluating: {}".format(category))
            # get cat_outputs and cat_answers
            try:
                cat_outputs = output_dict_w_cat[category]
                cat_answers = answer_dict_w_cat[category]
            except KeyError:
                print("Skipping {} for not found".format(category))
                continue

            exampels_to_eval = []
            for data_id, parsed_pred in cat_outputs.items():
                question_type = cat_answers[data_id]['question_type']
                if question_type != 'multiple-choice':
                    parsed_pred = parse_open_response(parsed_pred) # mainly for type consistency (make it number, etc.)
                else:
                    parsed_pred = parsed_pred

                exampels_to_eval.append({
                    "id": data_id,
                    "question_type": question_type,
                    "answer": cat_answers[data_id]['ground_truth'],
                    "parsed_pred": parsed_pred
                })

            judge_dict, metric_dict = evaluate(exampels_to_eval)
            metric_dict.update({"num_example": len(exampels_to_eval)})

            evaluation_result[category] = metric_dict

        scores = MMMUScore(evaluation_result)
        return scores
