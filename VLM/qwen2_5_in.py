from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json
import re
from huggingface_hub import login

# Login to Hugging Face if needed
# login("hf_your_token_here")  # Uncomment and paste your token here if accessing gated models

model_id = "google/flan-t5-xl"

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

# Prompt template
template = """Extract the object descriptions from the command, following the format:

Format:
<object> <description of the object>
Side: <left/right/middle/unspecified>
Distance: <closest/furthest/middle/unspecified>
Size: <big/small/medium/unspecified>

Example:
"Grab the smallest yellow object"
Output:
object small yellow object
Side: unspecified
Distance: unspecified
Size: small

Command: "{user_input}"


"""

def parse_model_output(raw_output: str):
    lines = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
    structured_objects = []
    i = 0

    while i < len(lines):
        match = re.match(r"([a-zA-Z]+)\s(.+)", lines[i])
        if match:
            obj_name, description = match.groups()
            side, distance, size = "unspecified", "unspecified", "unspecified"

            for j in range(i + 1, min(i + 4, len(lines))):
                if lines[j].startswith("Side:"):
                    side = lines[j].split(":", 1)[1].strip()
                elif lines[j].startswith("Distance:"):
                    distance = lines[j].split(":", 1)[1].strip()
                elif lines[j].startswith("Size:"):
                    size = lines[j].split(":", 1)[1].strip()

            structured_objects.append({
                "object": obj_name,
                "description": description,
                "side": side,
                "distance": distance,
                "size": size
            })
            i += 4
        else:
            i += 1

    # Deduplicate
    seen = set()
    deduped = []
    for obj in structured_objects:
        key = (obj['object'], obj['description'])
        if key not in seen:
            deduped.append(obj)
            seen.add(key)

    return deduped

# Main loop
print("FLAN-T5-XL is ready. Type your command (or 'exit' to quit):")

while True:
    user_input = input("Command: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting.")
        break

    prompt = template.format(user_input=user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating response...")
    output = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7, eos_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    structured = parse_model_output(response)

    if structured:
        print("\nParsed JSON:\n" + "-" * 30)
        print(json.dumps(structured, indent=2))
        print("-" * 30)
    else:
        print("No structured information extracted.")
