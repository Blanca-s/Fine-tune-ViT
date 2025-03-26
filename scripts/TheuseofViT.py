from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

image_path = '/224015062/ViT/airplane10.png'
image = Image.open(image_path)

processor = ViTImageProcessor.from_pretrained('/224015062/ViT/vit_cifar10_finetuned')
model = ViTForImageClassification.from_pretrained('/224015062/ViT/vit_cifar10_finetuned', output_attentions=True, num_labels=10)
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state

# attention_scores = outputs.attentions  # tuple of tensors, 每个tensor对应一层的注意力分数

# # 分析注意力分数
# for layer_idx, layer_attention in enumerate(attention_scores):
#    print(f"Layer {layer_idx} attention shape:", layer_attention.shape)
#    # shape: (batch_size, num_heads, sequence_length, sequence_length)
outputs = outputs.logits
predictions = outputs.argmax(dim=1)
print(predictions)
