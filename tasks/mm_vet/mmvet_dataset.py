from pathlib import Path

import utils
from tasks.base_dataset import Example, TaskDataset


class MMVetDataset(TaskDataset):
    """ MM-Vet Benchmark """
    def __init__(self, root, processor):
        # Note: we donot use templatizer and use default prompt format

        self.root = Path(root)
        self.data = utils.load(self.root / "mm-vet.json")

        self.processor = processor
        utils.print_rank_0(f"MM-Vet Benchmark total dataset size = {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Example:
        qid = f"v1_{index}"
        dic = self.data[qid]
        image_path = self.root / "images" / dic["imagename"]
        image = utils.load_image(image_path)
        question = dic["question"]
        answer = dic["answer"]

        # use default prompt as follows
        # The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.  {image_prompt}
        # Human: {question}
        # AI: """
        prompt = self.build_prompt(question)

        data = {
            "index": index,
            "prompt": prompt,
            "question": question,
            "question_id": qid,
            "answer": answer,
        }
        ex = Example(index, image, prompt, data)

        return ex
