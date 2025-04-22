# ROS2 Imports
import rclpy
from rclpy.node import Node
from rob8_interfaces.msg import LlmCommands

# LLM Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re

model_id = "Qwen/Qwen2.5-7B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Prompt template
template = """You are an information extraction engine that parses natural language commands into structured object descriptions.

Given a command, extract only the described objects and interpret their position, distance, and size.

Only return the extracted information in the following format — do not explain your reasoning, and do not output anything else.

Format:
<object> <description of the object>
Side: <left/right/middle/unspecified>
Distance: <closest/furthest/middle/unspecified>
Size: <big/small/medium/unspecified>

Example input:
"Place the large yellow cube to the right of the small green sphere."

Example output:
cube large yellow cube
Side: right
Distance: unspecified
Size: big

sphere small green sphere
Side: left
Distance: unspecified
Size: small

Now process this command:
"{user_input}"
"""

# ROS 2 publisher class
class ObjectInfoPublisher(Node):
    def __init__(self):
        super().__init__('object_info_publisher')
        self.publisher = self.create_publisher(LlmCommands, 'LLM_output', 10)

    def publish_info(self, data_list): 
        # Create a LlmCommands message 
        # and populate it with the structured data 
        # from the parsed output 
        objects = [] 
        descriptions = [] 
        sides = [] 
        distances = [] 
        sizes = [] 
        for item in data_list: 
            # Extract the object, description, side, distance, and size from the parsed output
            objects.append(item["object"]) 
            descriptions.append(item["description"]) 
            sides.append(item["side"]) 
            distances.append(item["distance"])  
            sizes.append(item["size"])  
        
        # Create a LlmCommands message and populate it with the structured data 
        LLM_model = LlmCommands()
        LLM_model.object = objects
        LLM_model.description = descriptions 
        LLM_model.side = sides
        LLM_model.distance = distances
        LLM_model.size = sizes
        
        self.publisher.publish(LLM_model)
        return 


# Parsing function
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

    seen = set()
    deduplicated_objects = []
    for obj in structured_objects:
        obj_key = (obj['object'], obj['description'])
        if obj_key not in seen:
            deduplicated_objects.append(obj)
            seen.add(obj_key)

    return deduplicated_objects


# Main interactive loop
def main():
    rclpy.init()
    publisher_node = ObjectInfoPublisher()

    print("Model is ready. Type your command (or 'exit' to quit):")

    try:
        while True:
            user_input = input("Command: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting.")
                break

            full_prompt = template.format(user_input=user_input)
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

            print("Generating response...")
            output = model.generate(**inputs, max_new_tokens=100, do_sample=False, eos_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(output[0], skip_special_tokens=True)

            # Trim to structured output only
            split_marker = "Now process this command:"
            if split_marker in response:
                response = response.split(split_marker, 1)[-1].strip()

            lines = response.splitlines()
            filtered_lines = [line.strip() for line in lines if "seconds to process" not in line and not line.startswith("Human:")]
            filtered_response = "\n".join(filtered_lines).strip()

            structured = parse_model_output(filtered_response)

            if structured:
                print("\nParsed JSON:\n" + "-"*30)
                print(json.dumps(structured, indent=2))
                print("-"*30)

                publisher_node.publish_info(structured)
            else:
                print("No structured information extracted.")

    except KeyboardInterrupt:
        pass

    publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

