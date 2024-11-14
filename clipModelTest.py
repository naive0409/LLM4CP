from PIL import Image
import json
import requests
from transformers import AutoProcessor, CLIPModel

model = CLIPModel.from_pretrained("./models/openai-clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("./models/openai-clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open('code_testing/000000039769.jpg')

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(logits_per_image)
print(probs)