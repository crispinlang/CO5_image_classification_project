from peft import LoraConfig, get_peft_model
from transformers import CLIPModel

def tuning():
    # GitHub PEFT: https://github.com/huggingface/peft
    # OpenClip model: https://huggingface.co/openai/clip-vit-base-patch16

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"] # these are the modules that need to be fine tuned for openCLIP modelss
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model
