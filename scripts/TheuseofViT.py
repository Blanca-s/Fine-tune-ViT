from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

image_path = 'C:\\Users\\28378\Desktop\Fine-tune-ViT\scripts\zooparty.png'
image = Image.open(image_path)

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-384")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-384")
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)

outputs = outputs.logits
print(outputs)
print(len(outputs[0]))

predictions = outputs.argmax(dim=1)
print(predictions)