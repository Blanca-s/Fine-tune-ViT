from transformers import ViTImageProcessor, ViTForImageClassification
from as_on_cifar_10 import AttentionScoreProcessor
import torch
import pickle
import numpy as np
from PIL import Image

processor = AttentionScoreProcessor("/224015062/ViT/vit_cifar10_finetuned")
all_attention_scores = processor.get_attention_scores("/224015062/ViT/test_batch")

def unpickle(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    # Unpack data
    images = batch[b'data']
    labels = batch[b'labels']
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels

inference_images, inference_labels = unpickle("/224015062/ViT/test_batch")

# Vision Transformer requires the input resolution to be 224x224, thus, we need to interpolate and enlarge those images
def resize_images(images, target_size=(224, 224)):
    resized_images = []
    for img in images:
        resized_img = Image.fromarray(img).resize(target_size, Image.BICUBIC)
        resized_images.append(np.array(resized_img))
    return np.array(resized_images)

# Implement this function on input images
resized_test_images = resize_images(inference_images)
print(f"Resized images shape: {resized_test_images.shape}")

# 检查是否有可用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

processor = ViTImageProcessor.from_pretrained("/224015062/ViT/vit_cifar10_finetuned")
model = ViTForImageClassification.from_pretrained(
    "/224015062/ViT/vit_cifar10_finetuned",
    output_attentions=True,
    attn_implementation="eager",
    num_labels=10
)

# 将模型移动到 GPU
model.to(device)

# Set the model to evaluation mode
model.eval()

# Batch setting
batch_size = 20

# Initialize the countings
correct = 0
total = 0

#  The number of data
total_samples = len(resized_test_images)  # Should be 10000 (CIFAR-10 test set size)
print(f"Total samples: {total_samples}")

# Process in batches
for i in range(0, len(resized_test_images), batch_size):
    batch_images = resized_test_images[i : i + batch_size]
    batch_labels = inference_labels[i: i + batch_size]
    batch_labels = torch.tensor(batch_labels).to(device)
    
    # Process the batch of images
    inputs = processor(images=list(batch_images), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs).logits
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == batch_labels).sum().item()
        total += batch_labels.size(0)

accuracy = correct / total
print(f"Model Accuracy on CIFAR-10 Test Set: {accuracy * 100:.2f}%")