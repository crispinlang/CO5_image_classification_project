from peft import LoraConfig, get_peft_model
from transformers import CLIPModel, Trainer, TrainingArguments
from src.preprocessing_rewrite import prepare_data
from src.clip_processing import clip_collator, clip_processor
from src.helpers import load_config

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

def fine_tuning(
    model_name="openai/clip-vit-base-patch32",
    output_dir="../model/mushroomCLIP",
    learning_rate=1e-3,
    num_train_epochs=2,
):

    cfg = load_config()
    data_cfg = cfg["data"]

    model = build_model(model_name=model_name)
    train_data, val_data, test_data, labels = prepare_data()
    processor = clip_processor(model_name=model_name)
    collator = clip_collator(processor=processor)
    model = build_model(model_name=model_name)
    

    #https://huggingface.co/docs/peft/main/quicktour
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=data_cfg["BATCH_SIZE"],
        per_device_eval_batch_size=data_cfg["BATCH_SIZE"],
        dataloader_num_workers=data_cfg["NUM_WORKERS"],
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False
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
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    return model, trainer, test_data