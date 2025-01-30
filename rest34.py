import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import torch.nn as nn
import torch.optim as optim

# CUDA Desteği
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Dataset Yolu
data_dir = r"C:\\Users\\berat\\PycharmProjects\\archive\\TuSimple\\train_set"
label_file = os.path.join(data_dir, "label_data_0601.json")

# TuSimple Dataset Sınıfı
class TuSimpleDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []

        # JSON Verisini Yükle
        with open(label_file, 'r') as f:
            self.data = [json.loads(line) for line in f]  # Her satırı ayrı JSON olarak işle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.data_dir, item['raw_file'])
        image = Image.open(img_path).convert("RGB")

        # Şerit bilgileri ve yüksekliği h_samples
        lanes = item['lanes']
        h_samples = item['h_samples']

        # Şerit Bilgilerini Piksel Maskesine Dönüştür
        mask = torch.zeros((224, 224), dtype=torch.float32)
        for lane in lanes:
            for x, y in zip(lane, h_samples):
                if x != -1:  # Şeridin olmadığı yerler -1 ile işaretlenmiş
                    x_rescaled = int(x * 224 / image.width)
                    y_rescaled = int(y * 224 / image.height)
                    mask[y_rescaled, x_rescaled] = 1  # Şeridi maskeye ekle

        if self.transform:
            image = self.transform(image)
        return image, mask  # Maskeyi [224, 224] boyutunda bırak

# Veri Dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Görselleri ResNet için ölçeklendir
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DataLoader
if __name__ == "__main__":
    dataset = TuSimpleDataset(data_dir, label_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)  # num_workers=0 (Windows için güvenli)

    # Model Tanımlama
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    # Son katmanı şerit algılama için özelleştirin
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 224 * 224),  # Çıkış: 224x224 piksel için tahmin
        nn.Sigmoid()  # Çıkışı [0, 1] aralığına sıkıştır
    )
    model = model.to(device)

    # Kayıp Fonksiyonu ve Optimizasyon
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss (piksel bazlı segmentasyon için uygun)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Eğitim Döngüsü
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # İleri Yayılım
            outputs = model(images)
            outputs = outputs.view(-1, 1, 224, 224)  # Çıkışı [batch, 1, 224, 224] boyutuna getir
            loss = criterion(outputs, masks.unsqueeze(1))  # Maskeyi [batch, 1, 224, 224] yap

            # Geri Yayılım
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    # Modeli Kaydet
    torch.save(model.state_dict(), "resnet34_tusimple_segmentation.pth")
    print("Model eğitimi tamamlandı.")
