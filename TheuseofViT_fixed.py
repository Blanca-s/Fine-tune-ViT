import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 加载 CIFAR-10 数据集
def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

# 数据集路径
data_dir = r'C:\Users\28378\Desktop\Fine-tune-ViT\cifar-10-batches-py'

# 加载训练批次
train_data = []
train_labels = []

for i in range(1, 6):
    batch_file = os.path.join(data_dir, f'data_batch_{i}')
    batch = unpickle(batch_file)
    train_data.append(batch[b'data'])
    train_labels.append(batch[b'labels'])

train_data = np.concatenate(train_data)
train_labels = np.concatenate(train_labels)

# 加载标签名称
meta_file = os.path.join(data_dir, 'batches.meta')
meta = unpickle(meta_file)
label_names = meta[b'label_names']

# 显示 CIFAR-10 数据集中的一张图像
image_data = train_data[0]  # 选择第一张图像
image_data = image_data.reshape(3, 32, 32).transpose(1, 2, 0)  # 重塑为 32x32x3

# 使用 PIL 将其转换为图像
image = Image.fromarray(image_data)

# 显示图像
plt.imshow(image)
plt.show()

# 处理和预测图像                    if you want to use fast processor,just as the warning ,you can change next raw as
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-384")    #processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-384", use_fast=True)
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-384")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# 获取预测结果
outputs = outputs.logits
print(outputs)
print(len(outputs[0]))

predictions = outputs.argmax(dim=1)
print(predictions)
