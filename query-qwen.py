from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

print(f"⚡ Loading {model_name} model locally...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"✅ Model {model_name} is ready!\n")

def query_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(inputs['input_ids'], max_length=512, num_return_sequences=1, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("🚀 CodeLlama Instruct Model Query — Type 'exit' to quit.\n")
    while True:
        user_input = input("📝 Enter your prompt: ")
        if user_input.lower() == "exit":
            print("👋 Exiting.")
            break
        
        print("⚙️  Querying model...")
        result = query_model(user_input)
        print("\n💡 Model response:\n")
        print(result)
        print("\n" + "="*80 + "\n")
