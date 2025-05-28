# ROS2 Imports
import rclpy
from rclpy.node import Node
from rob8_interfaces.msg import LlmCommands


# LLM Imports
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import re
from transformers import pipeline # NEW



model_id = "unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load model (already quantized to 4-bit with bitsandbytes)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # Automatically chooses GPU if available
    trust_remote_code=True,
)

# Build text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)



# Prompt template
template = """You are an assistant that helps locate objects and parses natural language commands into structured object descriptions.

Given a command that suggest that you should pick up the object, extract the described objects and action and interpret their position, distance, and size.

Return the extracted information in the following format. 

Format:
object: <object> 
description: <description of the object>
side: <left/right/middle/unspecified>
distance: <closest/furthest/middle/unspecified>
size: <big/small/medium/unspecified>

If no relevant information is given, respond politely with a helpful message, then fill in the structure with default placeholders as shown below.

Example input:
"Find the large yellow cube to the right."

Example output:
object: cube
description: yellow cube
side: right
distance: unspecified
size: big

Example input: 
"Hello, what can you do?" 

Example output:
"Hello, I am a helpful assistant robot that can help locate objects and hand them over. Is there any object you are looking for?"

object: non
description: non
side: unspecified
distance: unspecified
size: unspecified

Now process this command:
"{user_input}"

Response:
"""


# ROS 2 publisher class
class ObjectInfoPublisher(Node):
    def __init__(self):
        super().__init__('object_info_publisher')
        self.publisher = self.create_publisher(LlmCommands, '/LLM_output', 10)

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

            matches = re.findall(
                r"object:\s*(.*?)\s*description:\s*(.*?)\s*side:\s*(.*?)\s*distance:\s*(.*?)\s*size:\s*(.*?)(?=\nobject:|\Z)", 
                response, 
                flags=re.DOTALL
            )

            # Extract only the 4rd one if it exists
            if len(matches) >= 3:
                third = matches[4]
                clean_output = {
                    "object": third[0].strip(),
                    "description": third[1].strip(),
                    "side": third[2].strip(),
                    "distance": third[3].strip(),
                    "size": third[4].strip()
                }
                print(json.dumps(clean_output, indent=2))
                
                # Send to ROS
                publisher_node.publish_info([clean_output])
            else:
                print("Error")
                print(response)
            
            # Trim to structured output only
            split_marker = "Now process this command:"
            if split_marker in response:
                response = response.split(split_marker, 1)[-1].strip()

            #print(response)
            #publisher_node.publish_info(structured)
            
    except KeyboardInterrupt:
        pass

    publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
