# 2025/7/2
# zhangzhong
# https://huggingface.co/docs/peft/quicktour
# 这个chat给出了两种模型微调在数据处理上的对比
# https://chat.deepseek.com/a/chat/s/e2c9e0b6-d57d-4c8a-b7cc-8c122e58a377


# Train
# Each PEFT method is defined by a PeftConfig class that stores all the important parameters for building a PeftModel.
from peft import LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    # Causal LM
    # 好像默认的Arguments和Trainer就是为了CausalLM设计的？
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,  # for quantization
    #
    # seq2seq
    # AutoModelForSeq2SeqLM,
    # Seq2SeqTrainingArguments,
    # Seq2SeqTrainer,
    # DataCollatorForSeq2Seq,
)
from peft import get_peft_model
from datasets import load_dataset
import evaluate
import numpy as np

# https://discuss.pytorch.org/t/bfloat16-native-support/117155/6
# check this link and show 3090 support bf16

# dataset = load_dataset("wmt16", "ro-en")
# https://chat.deepseek.com/a/chat/s/dd253749-8ac5-4a65-9c49-6e3c60916ab3
# https://huggingface.co/datasets/yahma/alpaca-cleaned
dataset = load_dataset("yahma/alpaca-cleaned")

# model_name = "meta-llama/Llama-3.1-8B-Instruct"
# use a small model， 1B
model_name = "meta-llama/Llama-3.2-1B"
# TODO：微调完的模型要怎么部署起来？尤其我们是量化微调的？

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
# LLaMA and many open models:
# 	•	Do not have a default pad token.
# 	•	They were trained with the End-of-Sequence (eos_token) as their only special token.
#  End-of-Sequence token (which the model knows) as padding.
tokenizer.pad_token = tokenizer.eos_token  # Very important for causal LM


# 原来这样配置就成了QLoRA了
# quantization
bnb_config = BitsAndBytesConfig(
    # 4-bit quantization saves memory on the model side
    # Model Weights 4-bit quantized
    load_in_4bit=True,
    # Set precision of forward/backward computation
    # gradient, computation, optimizer is in floating point
    bnb_4bit_compute_dtype="bfloat16",  # or bfloat16 if you card supports it, >=A100, 3090
    # QLoRA quantization
    # double quantization.
    bnb_4bit_use_double_quant=True,
    # This selects the quantization algorithm.
    # default is fp4, regular 4bit floating point
    # nf4 stands for NormalFloat4, which is a special 4-bit floating-point format introduced in the QLoRA paper.
    bnb_4bit_quant_type="nf4",
)


# Since Llama is a causal/decoder-only model, use AutoModelForCausalLM:
# TODO: 咱们先不做量化加载试一下，毕竟咱们有四张3090
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # QLoRA quantization
    # device map is auto will automatically cut the model to different GPUs
    # but quantization only support single GPU
    # ValueError: You can't train a model that has been loaded in 8-bit or 4-bit precision on a different device than the one you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device()}` or `device_map={'':torch.xpu.current_device()}`
    #  device_map="auto",
)


# TODO:done read the lora paper
# task_type: the task to train for (sequence-to-sequence language modeling in this case)
# inference_mode: whether you’re using the model for inference or not
# r: the dimension of the low-rank matrices
# lora_alpha: the scaling factor for the low-rank matrices
# lora_dropout: the dropout probability of the LoRA layers

# TODO: read the qlora paper
# TODO: deploy the original model and instruct fine tune model


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


def preprocess_function(example):
    # 把
    if example["input"]:
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n\n{example['output']}"
    else:
        return f"### Instruction:\n{example['instruction']}\n\n### Response:\n\n{example['output']}"


# tokenize dataset
datasets = dataset.map(
    lambda x: {"text": preprocess_function(x)},
    # batched=True,
    # remove_columns=dataset["train"].column_names,  # remove original columns
    # remove_columns=["translation"],  # or you can specify the column to remove
    # desc="Running tokenizer on dataset",
    load_from_cache_file=True,
)


## ...
# 忘了tokenize了
# 用 tokenizer 把 text 转成 input_ids 和 labels
# 因为trainer需要这两个信息
# 又因为这是自回归语言模型 input_ids和labels是一样的
def tokenize_function(examples):
    # tokenizer(examples["text"], truncation=True, max_length=512)
    # 我这里没有做truncation，
    # return tokenizer(examples["text"])
    # result = tokenizer(examples["text"])
    # result["labels"] = result["input_ids"].copy()
    # return result

    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",  # # 必须静态 padding
        # 否则报错 ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (labels in this case) have excessive nesting (inputs type list where type int is expected).
    )
    result["labels"] = result["input_ids"].copy()
    return result


# TODO
# 这个地方还有一个关键的问题
# chatgpt告诉我需要同时这个 input_ids 和 labels 并且因为是自回归的，labels相比input_ids需要shift
# 但是我没有设置，代码反而成功运行了
# 我需要查看官方的例子，确认这一点！


# 又忘了做dynamic padding
# 可以在tokenize的时候做static padding，就是预先把所有example都pad到相同的长度
# 也可以在训练的时候，load数据的时候根据batch做dynamic padding

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=True,
)


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # For LLaMA, these are common
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# Data collator
# for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    # mlm=True, masked language modeling task, BERT
    # mlm=False, causal language modeling, GPT
    mlm=False,
    # 这里已经自动做了 dynamic padding
    # model=model,
    # 就没这个参数，，，
    # padding=True,  # set dynamic padding during training (True or "longest" for longest in batch, False for no padding)
)


# Training arguments
training_args = TrainingArguments(
    output_dir=f"./huggingface/lora/{model_name}-lora",
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    # 加了这么一些参数
    # num_train_epochs=2,
    gradient_accumulation_steps=16,  # 这个应该是加快训练速度的
    warmup_steps=100,
    max_steps=3000,
    # in the model setting, we use 4bit quantization
    # and in this training set, we use bf16
    # this told huggingface to use mixed precision training
    # fp16=True,  # Use mixed precision training
    bf16=True,
    # per_device_eval_batch_size=4,
    # weight_decay=0.01,
    # eval_strategy="epoch",
    save_strategy="steps",
    save_steps=500,  # Save every 500 steps
    # 因为没有eval 所以没有办法判断best model
    # 就微调完了就结束了
    # load_best_model_at_end=True,
    # predict_with_generate=True,  # Enable generation for evaluation
    logging_steps=10,  # Log every 10 steps
    # ⭐ 关键：必须显式告诉 Trainer labels 的名字，否则警告 No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
    label_names=["labels"],
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    # eval_dataset=tokenized_datasets["validation"],
    # tokenizer=tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained(f"./huggingface/lora/{model_name}-finetuned")
tokenizer.save_pretrained(f"./huggingface/lora/{model_name}-finetuned")
