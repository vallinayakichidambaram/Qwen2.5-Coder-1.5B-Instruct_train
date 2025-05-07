from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

dataset_path = "./dataset/solidity_data_instruct.txt"
dataset = load_dataset("text", data_files={"train": dataset_path})

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
model.config.use_sliding_window = False
print(model.config)
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=1)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./qwen-finetune",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=5e-5,
    fp16=False,  
    report_to="none",
    dataloader_pin_memory=False 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("./qwen-finetune")
tokenizer.save_pretrained("./qwen-finetune")

print("✅ Fine-tuning complete! Model saved to ./qwen-finetune")
