"""HF transformers has complicated structure, letting explicit converting checkpoint
for resolution interpolation easier than online surgery.
This script provides explicit converting of CLIP checkpoint for resolution interpolation.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings


def interpolate_pos_emb(pos_emb, scale, mode="bicubic"):
    """
    Args:
        pos_emb [n_patches, emb_dim]
        H, W: target shape
    """
    pe_cls, pe_grid = pos_emb[:1], pos_emb[1:]
    N = pos_emb.size(0) - 1
    size = int(N ** 0.5)
    # interpolate original size -> target size
    pe_grid = F.interpolate(
        rearrange(pe_grid, "(h w) d -> () d h w", h=size, w=size),
        scale_factor=scale,
        mode=mode,
        align_corners=False,
    ).squeeze(0)
    pe_grid = rearrange(pe_grid, "d th tw -> (th tw) d")
    pe = torch.cat([pe_cls, pe_grid])

    return pe


def surgery_clip_pos_emb_(model: CLIPVisionModel, target_size):
    """In-place surgery of resolution of CLIP position embedding via interpolation.
    """ 
    if target_size == model.config.image_size:
        return

    scale = target_size / model.config.image_size

    # update config
    model.config.image_size = target_size

    # interpolation
    org_emb = model.vision_model.embeddings
    interpolated_pos_emb = interpolate_pos_emb(org_emb.position_embedding.weight, scale)

    emb = CLIPVisionEmbeddings(model.config)
    emb.position_embedding.weight.detach().copy_(interpolated_pos_emb)

    for p1, p2 in zip(org_emb.parameters(), emb.parameters()):
        p2.requires_grad_(p1.requires_grad)

    model.vision_model.embeddings = emb


def convert_clip_resolution(from_ckpt: str, to_ckpt: str, size: int = 448):
    """Load CLIP checkpoint and convert resolution for interpolation.
    """
    print("Load model ...")
    model = CLIPVisionModel.from_pretrained(from_ckpt)

    print("Interpolate ...")
    scale = size / model.config.image_size

    # update config
    model.config.image_size = size

    # interpolation
    pos_emb = model.vision_model.embeddings.position_embedding
    interpolated_pos_emb = interpolate_pos_emb(pos_emb.weight, scale)

    emb = CLIPVisionEmbeddings(model.config)
    init_w = emb.position_embedding.weight.clone()
    emb.position_embedding.weight.detach().copy_(interpolated_pos_emb)
    model.vision_model.embeddings = emb

    # sanity check
    assert pos_emb.weight.requires_grad == emb.position_embedding.weight.requires_grad
    w = emb.position_embedding.weight
    print(f"\t[Org] Mean diff = {(pos_emb.weight.mean() - w.mean()).abs().item()}")
    print(f"\t[Org] Std diff = {(pos_emb.weight.std() - w.std()).abs().item()}")

    print(f"\t[Init] Mean diff = {(init_w.mean() - w.mean()).abs().item()}")
    print(f"\t[Init] Std diff = {(init_w.std() - w.std()).abs().item()}")

    print("Save ...")
    model.save_pretrained(to_ckpt)

    print("Done.")


if __name__ == "__main__":
    size = 448
    local_ckpt = "./hf_models/models--openai--clip-vit-large-patch14"
    target_path = f"./hf_models/models--openai--clip-vit-large-patch14-interpolate-{size}"
    convert_clip_resolution(local_ckpt, target_path, size)
