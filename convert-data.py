import os

# Extract Solidity files into plain text
source_dir = "./train/lido-contracts/"
raw_output_file = "./dataset/solidity_data.txt"

with open(raw_output_file, "w", encoding="utf-8") as out_file:
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".sol"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    code = f.read()
                    out_file.write(code + "\n\n")

print(f"✅ Extracted Solidity code into {raw_output_file}")

# Convert plain code to instruction-format dataset
instruct_output_file = "./dataset/solidity_data_instruct.txt"

# Ensure the dataset directory exists
os.makedirs(os.path.dirname(instruct_output_file), exist_ok=True)

with open(raw_output_file, "r", encoding="utf-8") as f_in, open(instruct_output_file, "w", encoding="utf-8") as f_out:
    for idx, code in enumerate(f_in.read().split("\n\n")):
        if code.strip() == "":
            continue
        f_out.write(f"### Instruction:\nExplain and display this Solidity contract.\n\n### Response:\n{code.strip()}\n\n")

print(f"✅ Converted data into instruction format at {instruct_output_file}")
