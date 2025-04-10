#'''
import torch # type: ignore
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
            {"type": "text", "text": "Identify all colors present in the image"},
            {"type": "text", "text": "Extract key attributes as [location, color, object]. If an attribute is missing, leave it blank."},
            {"type": "text", "text": "Example response: 'The green object is is a cube, and is located on the left side of the image. [50, 192]'"},
            {"type": "text", "text": "Example response: 'The red object is a sphere, and is located in the center of the image. The pixel coordinates are [200,120]'"},
            {"type": "text", "text": "If there are multiple objects, separate them with commas."},
            {"type": "text", "text": "If the image contains no objects, respond with 'No objects detected.'"},
        ],
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
        raw_image = raw_image.resize((256, 256))
        raw_image.show()  # Display the resized image for verification
        print(f"Image resized to: {raw_image.size}")
    except Exception as e:
        print(f"Error resizing image: {e}")
        return
    
    # Add the image and the command to the conversation
    prompt = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Now process this command: '{command}'."},
            {"type": "image", "image": raw_image}  # Add image directly here, not just the path
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
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
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
image_path = "/home/max/Documents/P8-project/color_0054.png"

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
