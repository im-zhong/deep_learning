# 2025/7/2
# zhangzhong
# https://huggingface.co/docs/peft/quicktour


# Train
# Each PEFT method is defined by a PeftConfig class that stores all the important parameters for building a PeftModel.
from peft import LoraConfig, TaskType
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_model

# TODO: read the lora paper
# task_type: the task to train for (sequence-to-sequence language modeling in this case)
# inference_mode: whether you’re using the model for inference or not
# r: the dimension of the low-rank matrices
# lora_alpha: the scaling factor for the low-rank matrices
# lora_dropout: the dropout probability of the LoRA layers

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# Once the LoraConfig is setup, create a PeftModel with the get_peft_model() function.
# need two things:
# 1. It takes a base model - which you can load from the Transformers library
# 2. the LoraConfig containing the parameters for how to configure a model for training with LoRA.


# default huggingface home is ~/.cache/huggingface
# the downloaded model and dataset will be saved in it
# you could configure the HF_HOME environment variable to change the path
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# trainable params: 2,359,296 || all params: 1,231,940,608 || trainable%: 0.1915
# Out of bigscience/mt0-large’s 1.2B parameters, you’re only training 0.19% of them!

# That is it 🎉! Now you can train the model with the Transformers Trainer, Accelerate, or any custom PyTorch training loop.
training_args = TrainingArguments(
    output_dir="./huggingface/lora/bigscience/mt0-large-lora",
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
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
