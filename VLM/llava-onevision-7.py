#'''
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


# Pre-Model Clean up
import gc
gc.collect()
torch.cuda.empty_cache()

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Llava model & processor
model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Initialize conversation history
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "You are a visual assistant that helps a robot understand which object a human is referring to based on an image and a list of bounding boxes."},
            {"type": "text", "text": "The robot needs to know which box best matches the command, based on features like color, size, shape, and position (e.g., leftmost, rightmost, smallest)."},
            {"type": "text", "text": "You will be given: 1. A human command, 2. An image, 3. A list of bounding boxes in the format: [x_min, y_min, x_max, y_max]"},
            {"type": "text", "text": "Your job is to examine all the bounding boxes in the image and choose the one that best matches the human's request."},
            {"type": "text", "text": "Respond with the number of the box that matches best, and explain why."},
            # The image would be passed along with this conversation
        ]
    }
]

def process_command(command, image_path):
    """Process command with the image added to the conversation."""
    
    # Load and display the image
    try:
        raw_image = Image.open(image_path)
        #raw_image.show()  # Display the image for verification
        raw_image = raw_image.convert("RGB")  # Convert to RGB just in case it's in another mode
        print("Image loaded and displayed.")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Resize the image to 256x256 (adjust if necessary)
    try:
        #raw_image = raw_image.resize((256, 256))
        raw_image.show()  # Display the resized image for verification
        print(f"Image resized to: {raw_image.size}")
    except Exception as e:
        print(f"Error resizing image: {e}")
        return
    

boundingbox_list = [
[853.93, 332.46, 967.83, 423.3],
[1027.75, 126.01, 1219.79, 268.15],
[390.53, 247.4, 457.16, 299.31]
]

# Construct bounding box description
box_description = "\n".join([f"Box {i+1}: {box}" for i, box in enumerate(boundingbox_list)])

# Extract box names (Box 1, Box 2, ...) for prompt constraint
valid_box_names = ", ".join([f"Box {i+1}" for i in range(len(boundingbox_list))])

from PIL import ImageDraw

def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        box = [int(coord) for coord in box]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]-10), f"Box {i+1}", fill="white")
    return image

# Draw boxes on the image
image_with_boxes = draw_boxes(raw_image.copy(), boundingbox_list)

# Add the image and structured command to the conversation
prompt = {
    "role": "user",
    "content": [
        {"type": "text", "text": f"A human says: '{command}'"},
        {"type": "text", "text": "These are the bounding boxes of objects in the image:"},
        {"type": "text", "text": box_description},
        {"type": "text", "text": f"Please analyze the image and respond with a sentence like: 'Box X. [Short explanation]'."},
        {"type": "text", "text": f"Only respond using one of the following: {valid_box_names}. Do not make up boxes."},
        {"type": "text", "text": "Use the area of each bounding box to determine size: (x_max - x_min) * (y_max - y_min)."},
        {"type": "image", "image": image_with_boxes}
    ]
}

# Update conversation
conversation.append(prompt)

# Apply chat template for formatting
try:
    formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    print("Formatted prompt created successfully.")
except Exception as e:
    print(f"Error formatting prompt: {e}")
    return

# Process inputs for the model
try:
    inputs = processor(images=image_with_boxes, text=formatted_prompt, return_tensors="pt")
    print("Inputs processed successfully.")
except Exception as e:
    print(f"Error processing inputs: {e}")
    return

# Convert float tensors to float16, but keep integer tensors unchanged
for key, value in inputs.items():
    if value.dtype == torch.float:
        inputs[key] = value.to(device, torch.float16)
    else:
        inputs[key] = value.to(device)  # Keep integer tensors in original dtype

# Generate output (no max token limit)
try:
    with torch.no_grad():
        output = model.generate(**inputs,max_new_tokens=200)
    print("Output generated successfully.")
except Exception as e:
    print(f"Error generating output: {e}")
    return

# Decode result
try:
    extracted_colors = processor.decode(output[0][2:], skip_special_tokens=True)
    print(f"Decoded output: {extracted_colors}")
except Exception as e:
    print(f"Error decoding output: {e}")
    return

print(f"\n🔹 Extracted Colors: {extracted_colors}\n")
return extracted_colors

# Load the image once before the loop
image_path = "./color_0946.png"

# Example usage to test color identification 
while True:  
    print("Ready for task:")
    prompt = input()  # Take user command
    process_command(prompt, image_path)  # Process the command with the image in the conversation
