import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.optim as optim

# استخدام نموذج مدرب مسبقًا (Pretrained Model)
class CNN_1(nn.Module):
    def __init__(self, input_size, n_feature, output_size):
        super(CNN_1, self).__init__()
        
        # استخدام ResNet18 المدرب مسبقًا
        self.resnet = models.resnet18(pretrained=True)  # استخدام ResNet18 المدرب مسبقًا
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_size)  # تعديل الطبقة النهائية لتناسب المهمة

    def forward(self, x):
        x = self.resnet(x)
        return x

# تجهيز البيانات مع Augmentation إضافي
transform = transforms.Compose([
    transforms.RandomResizedCrop(128),  # Crop عشوائي
    transforms.RandomHorizontalFlip(),  # Flip عشوائي
    transforms.RandomRotation(20),     # دوران عشوائي للصورة
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # تعديل الألوان عشوائيًا
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# تحميل البيانات
train_dataset = datasets.ImageFolder(root='Sample1', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# تهيئة النموذج والمعلمات
input_size = (3, 128, 128)
n_feature = 16
output_size = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_1(input_size, n_feature, output_size).to(device)

# تحسين المعدلات باستخدام AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# تطبيق Learning Rate Scheduler
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=10)

# تدريب النموذج
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # حساب التوقعات الصحيحة
        _, predicted = torch.max(outputs, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()

    accuracy = 100 * correct_preds / total_preds
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
    
    # تطبيق الـ Scheduler لتقليل معدل التعلم
    scheduler.step()

# حفظ الأوزان الجديدة
torch.save(model.state_dict(), "cnn1_model_improved.pth")
print("Improved model weights saved as cnn1_model_improved.pth")
