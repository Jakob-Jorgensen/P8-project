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
            {"type": "text", "text": "You will be given:"},
            {"type": "text", "text": "1. A human command"},
            {"type": "text", "text": "2. An image"},
            {"type": "text", "text": "3. A list of bounding boxes in the format: [x_min, y_min, x_max, y_max]"},
            {"type": "text", "text": "Your job is to examine all the bounding boxes in the image and choose the one that best matches the human's request."},
            {"type": "text", "text": "Respond with the number of the box that matches best, and explain why."},
            {"type": "text", "text": "Example:"},
            {"type": "text", "text": "Command: 'Get me the rightmost red object.'"},
            {"type": "text", "text": "Bounding boxes: [[30,40,80,90],[150,50,190,130],[23,16,76,94]]"},
            {"type": "text", "text": "Response: 'Box 2. It is the rightmost object and it is red.'"},
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
    

    boundingbox_list = [[853.93, 332.46, 967.83, 423.3],[1027.75, 126.01, 1219.79, 268.15],[390.53, 247.4, 457.16, 299.31]]
    # Add the image and the command to the conversation
    prompt = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Now process this command: '{command}'."},
            {"type": "text", "text": f"Bounding boxes: '{boundingbox_list}'."},
            {"type": "image", "image": raw_image}  # Add image directly here, not just the path
        ]
    }
 # Bounding boxes:\nBox 1: [10, 30, 60, 80]\nBox 2: [200, 40, 260, 100]


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
        inputs = processor(images=raw_image, text=formatted_prompt, return_tensors="pt")
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

'''
import json
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Llava model & processor
model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Initialize conversation history
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "You are assisting a robot in identifying and manipulating objects based on visual input and natural language commands."},
            {"type": "text", "text": "Extract key attributes as [position, color, object]. If an attribute is missing, leave it blank."},
            {"type": "text", "text": "Example: 'Take the rightmost blue cube' → rightmost, blue, cube"},
        ],
    }
]

def add_new_command(command, image_path):
    """Dynamically update the conversation with a new command + image."""
    new_prompt = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Now process this command: '{command}'."},
            {"type": "image"}
        ],
    }
    conversation.append(new_prompt)  # Add new command + image placeholder
    
    # Load image
    raw_image = Image.open(image_path)

    # Apply chat template for formatting
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Process inputs for the model
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt")

    # Convert float tensors to float16, but keep integer tensors unchanged
    for key, value in inputs.items():
        if value.dtype == torch.float:
            inputs[key] = value.to(device, torch.float16)
        else:
            inputs[key] = value.to(device)  # Keep integer tensors in original dtype


    # Generate output
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    # Decode result
    extracted_attributes = processor.decode(output[0][2:], skip_special_tokens=True)
    
    print(f"\n🔹 Extracted Attributes: {extracted_attributes}\n")
    return extracted_attributes

# Example usage
add_new_command("What is in the image", "/home/marcus/CloudStation/Skole/Uni/8 Semester/Project/llava-vision/color_0054.png")
#add_new_command("Pick up the left blue block", "/home/mimi/Downloads/block.png")

# (Optional) Save conversation history for persistence
with open("conversation_history.json", "w") as f:
    json.dump(conversation, f)



#'''

