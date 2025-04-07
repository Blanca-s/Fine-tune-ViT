from transformers import ViTImageProcessor, ViTForImageClassification
from as_on_cifar_10 import AttentionScoreProcessor
import torch
import pickle
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from sklearn.metrics import accuracy_score
from PIL import Image

# 处理 ViT 的 Attention Scores
processor = AttentionScoreProcessor("/224015062/ViT")
all_attention_scores = processor.get_attention_scores("/224015062/ViT/test_batch")

# 从 pickle 文件中加载数据（CIFAR-10 数据集）
def unpickle(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    # Unpack data
    images = batch[b'data']
    labels = batch[b'labels']
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)          # 调整图像形状，使其符合输入要求（CIFAR-10 图像的形状是 32x32）
    return images, labels

inference_images, inference_labels = unpickle("/224015062/ViT/test_batch")

# 检查是否有可用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

processor = ViTImageProcessor.from_pretrained("/224015062/ViT")
model = ViTForImageClassification.from_pretrained(
    "/224015062/ViT",
    output_attentions=True,
    attn_implementation="eager",
    num_labels=10
)

# 将模型移动到 GPU
model.to(device)

# 定义 CIFAR-10 数据集类
class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 将 numpy.ndarray 转换为 PIL.Image
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        return image, label
    
# A pipeline used to pre-process the data.
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# Load the dataset
train_images, train_labels = unpickle("/224015062/ViT/data_batch_1")
test_images, test_labels = unpickle("/224015062/ViT/test_batch")

train_dataset = CIFAR10Dataset(train_images, train_labels, transform=transform)
test_dataset = CIFAR10Dataset(test_images, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Freeze the backbone parameters
for param in model.vit.embeddings.parameters():
    param.requires_grad = False
for param in model.vit.encoder.parameters():
    param.requires_grad = False

# Optimizer and loss
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Train function
def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images).logits

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Evaluate function
# 模型评估
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 训练和评估
train_model(model, train_loader, criterion, optimizer, device, epochs=5)
evaluate_model(model, test_loader, device)

# Save the fine-tuned model
save_directory = "/224015062/ViT/vit_cifar10_finetuned"
model.save_pretrained(save_directory)
