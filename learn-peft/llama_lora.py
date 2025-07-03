# 2025/7/2
# zhangzhong
# https://huggingface.co/docs/peft/quicktour


# Train
# Each PEFT method is defined by a PeftConfig class that stores all the important parameters for building a PeftModel.
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model
from datasets import load_dataset
import evaluate
import numpy as np

dataset = load_dataset("wmt16", "ro-en")
model_name = "bigscience/mt0-large"
# Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json.
# Access to model meta-llama/Llama-3.1-8B-Instruct is restricted. You must have access to it and be authenticated to access it. Please log in.
# 用llama的模型还得先登录
# https://huggingface.co/docs/huggingface_hub/en/guides/cli
# 1. uv add "huggingface_hub[cli]" 用uv add装上竟然没用，必须是用 pip install -U "huggingface_hub[cli]"
# 大概率是因为uv add装的bin文件没有加到PATH里
# 2. huggingface-cli login
# 3. to https://huggingface.co/settings/tokens to get a token, and paste it in the terminal when prompted
# your are done
# then, you need to grant the access to the model
# Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json.
# Access to model meta-llama/Llama-3.1-8B-Instruct is restricted and you are not in the authorized list. Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct to ask for access.
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# TODO: read the lora paper
# task_type: the task to train for (sequence-to-sequence language modeling in this case)
# inference_mode: whether you’re using the model for inference or not
# r: the dimension of the low-rank matrices
# lora_alpha: the scaling factor for the low-rank matrices
# lora_dropout: the dropout probability of the LoRA layers


# Once the LoraConfig is setup, create a PeftModel with the get_peft_model() function.
# need two things:
# 1. It takes a base model - which you can load from the Transformers library
# 2. the LoraConfig containing the parameters for how to configure a model for training with LoRA.


# default huggingface home is ~/.cache/huggingface
# the downloaded model and dataset will be saved in it
# you could configure the HF_HOME environment variable to change the path
# model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")


# trainable params: 2,359,296 || all params: 1,231,940,608 || trainable%: 0.1915
# Out of bigscience/mt0-large’s 1.2B parameters, you’re only training 0.19% of them!

# That is it 🎉! Now you can train the model with the Transformers Trainer, Accelerate, or any custom PyTorch training loop.
# training_args = TrainingArguments(
#     output_dir="./huggingface/lora/bigscience/mt0-large-lora",
#     learning_rate=1e-3,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     num_train_epochs=2,
#     weight_decay=0.01,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
# )


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# trainer.train()


# inputs = [ex["translation"]["ro"] for ex in dataset["train"]]
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# # 1. Padding Options
# # "longest" (default): Pad to the longest sequence in the batch
# # "max_length": Pad to a specified maximum length (requires max_length parameter)
# # "do_not_pad" or False: No padding applied
# #
# # padding="max_length": All sequences will be padded to exactly 64 tokens
# # max_length=64: The target length for padding/truncation
# # truncation=True: Sequences longer than 64 tokens will be cut off
# #
# # 加上pad这个函数实在是太慢了，我还是在训练的时候dynamic padding吧
# tokenizer(inputs, max_length=128, truncation=True, padding=False)


def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["ro"] for ex in examples["translation"]]

    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
        padding=False,  # Dynamic padding to longest in batch (set to False for no padding during preprocessing
    )

    # tokenizer.as_target_tokenizer() is crucial for seq2seq models because:
    # Some tokenizers behave differently for source vs target text
    # For T5/mT5 models, it ensures proper handling of decoder inputs
    # It may add special tokens or handle BOS/EOS tokens differently
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding=False,  # Dynamic padding to longest in batch
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# tokenize dataset
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,  # remove original columns
    # remove_columns=["translation"],  # or you can specify the column to remove
    desc="Running tokenizer on dataset",
    load_from_cache_file=True,
)


peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# Data collator
# for dynamic padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,  # set dynamic padding during training (True or "longest" for longest in batch, False for no padding)
)

# 评估就不看了，毕竟我也不做翻译
# Evaluation metric
metric = evaluate.load("sacrebleu")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU score
    result = metric.compute(
        predictions=decoded_preds, references=[[ref] for ref in decoded_labels]
    )
    return {"bleu": result["score"]}


# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./huggingface/lora/{model_name}-lora",
    learning_rate=1e-3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    predict_with_generate=True,  # Enable generation for evaluation
    logging_steps=10,  # Log every 10 steps
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
trainer.save_pretrained(f"./huggingface/lora/{model_name}-finetuned")
