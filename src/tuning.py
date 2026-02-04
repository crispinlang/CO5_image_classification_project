from peft import LoraConfig, get_peft_model
from transformers import CLIPModel, Trainer

def build_model():
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

def fine_tuning():
    #https://huggingface.co/docs/peft/main/quicktour
    training_args = TrainingArguments(
        output_dir="../model/mushroomCLIP",
        learning_rate=1e-3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained("output_dir")