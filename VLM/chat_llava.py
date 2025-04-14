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

# Function to draw bounding boxes
def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        box = [int(coord) for coord in box]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]-10), f"Box {i+1}", fill="white")
    return image

# Main function to process the command
def process_command(command, image_path, boundingbox_list):
    try:
        raw_image = Image.open(image_path).convert("RGB")
        print("Image loaded.")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Annotate image with boxes
    image_with_boxes = draw_boxes(raw_image.copy(), boundingbox_list)

    # Prepare simple prompt
    box_description = "\n".join([f"Box {i+1}: {box}" for i, box in enumerate(boundingbox_list)])
    valid_box_names = ", ".join([f"Box {i+1}" for i in range(len(boundingbox_list))])

    prompt = (
        f"Command: '{command}'\n"
        f"Bounding boxes:\n{box_description}\n"
        f"Choose the best box ({valid_box_names}) and explain your choice."
    )

    # Format for model
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_with_boxes},
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
        inputs = processor(images=image_with_boxes, text=formatted_prompt, return_tensors="pt")
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

# Example bounding boxes and image
boundingbox_list = [
    [853.93, 332.46, 967.83, 423.3],
    [1027.75, 126.01, 1219.79, 268.15],
    [390.53, 247.4, 457.16, 299.31]
]
image_path = "./color_0946.png"

# Main loop
while True:
    print("Ready for command (type 'exit' to quit):")
    user_command = input()
    if user_command.lower() == "exit":
        break
    process_command(user_command, image_path, boundingbox_list)
