from dataclasses import dataclass

from PIL import Image
from torch.utils.data import Dataset

from pipeline.data_utils.templates import Templatizer


@dataclass
class Example:
    id: int  # results will be sorted by id
    image: Image
    prompt: str
    data: dict  # answer and additional data -- will be included in results


@dataclass
class Batch:
    ids: list[int]
    inputs: dict
    data: list[dict]


class TaskDataset(Dataset):
    def set_templatizer(self, dset_name: str, template_name: str):
        if template_name is None:
            self.templatizer = None
            return

        templatizer = Templatizer.from_names(template_name, dset_name)
        self.templatizer = templatizer

    def build_prompt(self, question, image_prompt="Human: <image>"):
        templatizer = getattr(self, "templatizer", None)
        if templatizer is not None:
            if type(question) == str:
                examples = [{"question": question}]
            elif type(question) == dict:
                examples = [question]
            else:
                raise NotImplementedError()
            prompt = templatizer(examples, image_prompt=image_prompt)
            return prompt

        prompt = f"""The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
{image_prompt}
Human: {question}
AI: """
        return prompt

    def collate_fn(self, examples: list[Example]) -> Batch:
        ids = [ex.id for ex in examples]
        data = [ex.data for ex in examples]

        images = [ex.image for ex in examples]
        prompts = [ex.prompt for ex in examples]
        inputs = self.processor(images=images, texts=prompts)

        batch = Batch(ids=ids, inputs=inputs, data=data)
        return batch

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int) -> Example:
        raise NotImplementedError()
