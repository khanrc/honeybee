import torch


def batchify(batch, tokenizer, max_length: int, use_trunc=False):
    """collate_fn
    Args:
        batch
        tokenizer
        max_length (int)
        use_trunc (bool)

    NOTE data["image"] can be None (e.g., text-instruction dataset)
    NOTE every batch for each device SHOULD have one image at least;
        if all batch data are text-only ones, error would occurs.
    """
    output_batch = {}
    image_list = [data["image"] for data in batch]

    # NOTE each image sample has "single image" (i.e., every sample has only 0 or 1 image):
    # - image data is a single image (not support multiple/interleaved images)
    num_images_per_sample = torch.LongTensor([img is not None for img in image_list])

    # 1. remove None images from image_list
    image_list = [img for img in image_list if img is not None]

    # 2. collate for images: [num_images, c, h, w]
    images = [img for img in image_list if img.shape[0] == 1]
    image_tensor = torch.cat(images) if images else None
    output_batch["pixel_values"] = image_tensor

    # 3. collate for text
    text_batch = [data["text"] for data in batch]
    padding = "longest" if use_trunc else "max_length"
    text_batch = tokenizer.batch_collate_pad(
        text_batch,
        padding=padding,
        padding_side="right",
        max_length=max_length,
    )

    # NOTE [bw-compat] Do not use attention mask for training, it will be generated automatically.
    text_batch.pop("attention_mask")

    output_batch.update({
        **text_batch,
        "num_images": num_images_per_sample,
    })

    return output_batch
