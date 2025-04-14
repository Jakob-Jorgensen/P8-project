import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import gc

# Clean up memory
gc.collect()
torch.cuda.empty_cache()

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and processor
model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id)


# Main function to process the command
def process_command(command, image_path):
    try:
        raw_image = Image.open(image_path).convert("RGB")
        print(f'Raw Image Size: {raw_image.size}')
        raw_image = raw_image.resize((256, 256))
        print("Image loaded.")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    prompt = (
        f"Command: '{command}'\n"
        f"Where is this object relative to the other objects, and what is its size?.\n"
    )

    # Format for model
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_image},
                {"type": "text", "text": command},
                {"type": "text", "text": prompt},
            ]
        }
    ]

    try:
        formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    except Exception as e:
        print(f"Error formatting prompt: {e}")
        return

    try:
        inputs = processor(images=raw_image, text=formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(device, torch.float16 if v.dtype == torch.float else None) for k, v in inputs.items()}
    except Exception as e:
        print(f"Error processing inputs: {e}")
        return

    try:
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=200)
        decoded = processor.decode(output[0], skip_special_tokens=True)
        print(f"\n🔍 Model Response:\n{decoded}\n")
        return decoded
    except Exception as e:
        print(f"Error during generation: {e}")

image_path = "./color_0946.png"

# Main loop
while True:
    print("Ready for command (type 'exit' to quit):")
    user_command = input()
    if user_command.lower() == "exit":
        break
    process_command(user_command, image_path)
