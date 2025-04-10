import requests

import torch
import cv2 as cv
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-base"
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_path = "./color_0946.png"
image = Image.open(image_path)
image = image.convert("RGB")
#image = image.resize((512,512))
# Check for cats and remote controls
text_labels = [["yellow box"]]

inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.3,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

image_cv = cv.imread(image_path)
#image_cv = cv.resize(image_cv, (512,512), interpolation=cv.INTER_LINEAR)

result = results[0]
print(result)
for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
    box = [round(x, 2) for x in box.tolist()]
    print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
    print(box[0])
    image_result = cv.rectangle(image_cv, (int(box[2]),int(box[3])), (int(box[0]),int(box[1])), (0,255,0),1)
cv.imshow("result", image_result)
cv.waitKey(0)
