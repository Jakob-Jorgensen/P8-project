# ROS2 Imports
import rclpy
from rclpy.node import Node
from rob8_interfaces.msg import LlmCommands
from std_msgs.msg import String
# LLM Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
import time 

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

Given a command that requires you to manipulate an object, extract only the desired objects and interpret their position, distance, and size.

Only return the extracted information in the following format. Do not explain your reasoning, and do not output anything else.
If no relevant object is given, respond with the object non.

Format:
object: <object> 
description: <description of the object>
side: <left/right/middle/unspecified>
distance: <closest/furthest/middle/unspecified>
size: <big/small/medium/unspecified>

Example input:
"Find the large yellow cube to the right."

Example output:
object: cube
description: yellow cube
side: right
distance: unspecified
size: big

Example input:
"Furthest away, next to the blue block there is a orange object, grab it."

Example output:
object: object
description: orange object
side: unspecified
distance: furthest
size: unspecified

Input:
{user_input}

Output:
"""

# ROS 2 publisher class
class ObjectInfoPublisher(Node):
    def __init__(self):
        super().__init__('object_info_publisher')
        self.publisher = self.create_publisher(LlmCommands, '/LLM_output', 10) 
        self.subscription = self.create_subscription(String,'human_command', self.msg_decrypt,10)
        self.UX_handling = self.create_publisher(String,'/aaulab_output', 10)
    # Function for running the speech2text into the LLM 
    def msg_decrypt(self,msg):
        user_input = msg.data.strip() 
        start_time = time.time()
        print(f"Prompt input: {user_input}")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting.")
            exit()

        full_prompt = template.format(user_input=user_input)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device) 
        #print(model.device)
        
        print("Generating response...")
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        matches = re.findall(
            r"object:\s*(.*?)\s*description:\s*(.*?)\s*side:\s*(.*?)\s*distance:\s*(.*?)\s*size:\s*(.*?)(?=\nobject:|\Z)", 
            response, 
            flags=re.DOTALL
        )

        # Extract only the 4rd one if it exists
        if len(matches) >= 4:
            third = matches[3]
            clean_output = {
                "object": third[0].strip(),
                "description": third[1].strip(),
                "side": third[2].strip(),
                "distance": third[3].strip(),
                "size": third[4].strip()
            }
            print(json.dumps(clean_output, indent=2)) 
            end_time = time.time() 
            elapsed_time = end_time - start_time
            print(f"Elapsed Time: {elapsed_time}")
            # Send to ROS
            self.publish_info([clean_output])
        
        # Trim to structured output only
        split_marker = "Now process this command:"
        if split_marker in response:
            response = response.split(split_marker, 1)[-1].strip()


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
        
        if LLM_model.object == LLM_model.description == LLM_model.side == LLM_model.distance == LLM_model.size =="non": 

            #self.get_logger().info("No objects was found.")  
            error_msg = String()
            error_msg.data = "Sorry I did not understand that please ask for an object that is on the table." 
            self.UX_handling.publish(error_msg)
            return

        else:
            self.publisher.publish(LLM_model)
            return 


# Main interactive loop
def main():
    try: 
        rclpy.init() 
        
        publisher_node = ObjectInfoPublisher()  
        
        print("Model is ready. Type your command (or 'exit' to quit):") 
        rclpy.spin(publisher_node)
            
    except KeyboardInterrupt:
        pass

    publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
