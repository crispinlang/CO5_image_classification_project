from transformers import CLIPProcessor

# https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPProcessor
# CLIPProcessor should simplify the pipeline (i hope?) becauseit wraps the image processor and tokenizer into one function.
def clip_processor(model_name="openai/clip-vit-base-patch32"):
    return CLIPProcessor.from_pretrained(model_name) 

def clip_collator(processor, return_loss=True):
    def collate_fn(examples):
        images = []
        texts = []

        for item in examples:
            if not isinstance(item, dict):
                raise ValueError("Expected dict samples with 'image' and 'text' keys.")
            if "image" not in item or "text" not in item:
                raise ValueError("Sample missing required 'image' or 'text' field.")

            image = item["image"]
            text = item["text"]

            images.append(image)
            texts.append(text)

        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if return_loss:
            # CLIPModel returns contrastive loss when this flag is present.
            batch["return_loss"] = True

        return batch

    return collate_fn
