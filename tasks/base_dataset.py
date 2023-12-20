from dataclasses import dataclass

from PIL import Image
from torch.utils.data import Dataset
from pipeline.data_utils.special_tokens import SYSTEM, HUMAN, AI

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
    def build_prompt(self, question, image_prompt="Human: <image>"):
        prompt = f"""{SYSTEM}
{image_prompt}
Human: {question}
AI: """
        return prompt

    def collate_fn(self, examples: list[Example]) -> Batch:
        ids = [ex.id for ex in examples]
        data = [ex.data for ex in examples]

        images = [ex.image for ex in examples]
        prompts = [ex.prompt for ex in examples]
        inputs = self.processor(images=images, text=prompts)

        batch = Batch(ids=ids, inputs=inputs, data=data)
        return batch
