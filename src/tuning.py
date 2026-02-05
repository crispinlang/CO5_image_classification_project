from peft import LoraConfig, get_peft_model
from transformers import CLIPModel, Trainer, TrainingArguments
from preprocessing_rewrite import prepare_data
from clip_processing import clip_collator, clip_processor

def build_model(model_name="openai/clip-vit-base-patch32"):
    # GitHub PEFT: https://github.com/huggingface/peft
    # OpenClip model: https://huggingface.co/openai/clip-vit-base-patch16

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )

    model = CLIPModel.from_pretrained(model_name)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def fine_tuning(model_name="openai/clip-vit-base-patch32"):

    model = build_model(model_name=model_name)
    train_data, val_data, test_data, labels = prepare_data()
    processor = clip_processor(model_name=model_name)
    collator = clip_collator() #tbd
    model = build_model(model_name=model_name)
    

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
        train_dataset=train_data, 
        eval_dataset=val_data,
        # not needed with CLIPProcessor:
        # processing_class=tokenizer,
        data_collator=collator,
        # ignoring for now:
        # compute_metrics=compute_metrics,
    )

    trainer.train()
