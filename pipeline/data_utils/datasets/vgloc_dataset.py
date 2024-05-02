import os.path as osp
from pathlib import Path

from .base_task import BaseTaskDataset
import utils


class VGLocDataset(BaseTaskDataset):
    """Visual Genome localization dataset"""
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        meta_root_path: str,
        image_root_path: str,
        dedup,
        nturn=1,  # number of QA pairs per image
        bbox_coord_style=3,
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, **kwargs)

        self.nturn = nturn
        self.bbox_coord_style = bbox_coord_style
        self.dedup = dedup
        self.dataset = self.load_data(meta_root_path, image_root_path)

        print(f"Load VGLoc dataset. len: {len(self.dataset)}")

    def load_meta(self, path: str | Path):
        js = utils.load(path)
        id_key = "image_id" if "image_id" in js[0] else "id"
        meta = {}
        for dic in js:
            image_id = dic.pop(id_key)
            meta[image_id] = dic
        return meta

    def load_data(self, meta_root_path, image_root_path):
        meta_root_path = Path(meta_root_path)
        images = utils.load(meta_root_path / "image_data.json")
        images = {
            dic["image_id"]: dic
            for dic in images
        }
        regions = utils.load(meta_root_path / "region_descriptions.json")

        data = []
        for dic in regions:
            image_id = dic["id"]
            region_list = dic["regions"]
            image_path = osp.join(image_root_path, f"{image_id}.jpg")
            for rdic in region_list:
                assert image_id == rdic["image_id"]
                x = rdic["x"]
                y = rdic["y"]
                w = rdic["width"]
                h = rdic["height"]

                image_dic = images[image_id]
                W = image_dic["width"]
                H = image_dic["height"]

                # formatting bbox
                bbox_str = self.preprocess_bbox(
                    x, y, w, h, W, H,
                    bbox_coord_style=self.bbox_coord_style,
                )

                phrase = rdic["phrase"]

                data.append((image_path, {"phrase": phrase, "bbox": bbox_str}))

        data = self.finalize_data(data, task_type="vgloc_loc", nturn=self.nturn)

        return data
