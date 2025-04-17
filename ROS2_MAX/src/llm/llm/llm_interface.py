# ROS2 Imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, List

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

        self.object_pub = self.create_publisher(String, 'object', 10)
        self.description_pub = self.create_publisher(String, 'description', 10)
        self.side_pub = self.create_publisher(String, 'side', 10)
        self.distance_pub = self.create_publisher(String, 'distance', 10)
        self.size_pub = self.create_publisher(String, 'size', 10)

    def publish_info(self, data_list):
        for item in data_list:
            object_msg = String()
            object_msg.data = item["object"]
            self.object_pub.publish(object_msg)

            desc_msg = String()
            desc_msg.data = item["description"]
            self.description_pub.publish(desc_msg)

            side_msg = String()
            side_msg.data = item["side"]
            self.side_pub.publish(side_msg)

            distance_msg = String()
            distance_msg.data = item["distance"]
            self.distance_pub.publish(distance_msg)

            size_msg = String()
            size_msg.data = item["size"]
            self.size_pub.publish(size_msg)

            self.get_logger().info(f"Published object: {item['object']}, Side: {item['side']}, Distance: {item['distance']}, Size: {item['size']}")
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

