from tasks.base_task import Task, TaskScore


class OWLScore(TaskScore):
    def get_summary(self, max_level=1):
        return "No quantitative evaluation for OWL-Eval."

    def dumps(self):
        return "No quantitative evaluation for OWL-Eval."


class OWLTask(Task):
    def compute_score(self, results):
        # dummy score for this dataset.
        score = OWLScore({"acc": 0.0})

        return score

    def reformat_results(self, results):
        """Result writing function for this Owl-Eval dataset.

        Args:
            results (list of dict): Prediction results with other meta informations
        """

        # Parse result for the structured formant.
        parsed_result_dict = []
        for item in results.values():
            temp_dict = {
                "image": item["image_path"],
                "question_id": item["question_id"],
                "question": item["question"],
                "answer": item["pred"],
                "model_id": "Custom_OWL",
            }
            parsed_result_dict.append(temp_dict)

        return parsed_result_dict
