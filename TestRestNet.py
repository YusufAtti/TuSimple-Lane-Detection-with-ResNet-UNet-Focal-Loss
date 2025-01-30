import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn  # Eksik olan import eklendi

# CUDA Desteği
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on: {device}")

# Eğitilmiş modeli yükleme
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 224 * 224),  # Çıkış: 224x224 piksel
    nn.Sigmoid()  # [0, 1] arasında sonuçlar
)
model.load_state_dict(torch.load("resnet34_tusimple_segmentation.pth"))
model = model.to(device)
model.eval()  # Tahmin moduna geçiş

# Dönüşüm (görüntüleri model için hazırlama)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Görüntüyü modele uygun boyuta getir
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test görüntülerinin bulunduğu klasör (belirtilen klasör)
test_images_dir = "C:/dataset_path/archive/TuSimple/test_set/clips/0530/1492626224112349377_0"

# Belirtilen klasördeki görüntüleri işleyin
for file_name in os.listdir(test_images_dir):
    if file_name.endswith(".jpg"):  # Yalnızca JPEG görüntüler
        image_path = os.path.join(test_images_dir, file_name)
        test_image = Image.open(image_path).convert("RGB")
        input_tensor = transform(test_image).unsqueeze(0).to(device)  # Görüntüyü modele hazırlayın

        # Model tahmini
        with torch.no_grad():
            output = model(input_tensor)
            output = output.view(224, 224).cpu().numpy()  # Çıkışı [224, 224] boyutuna getir

        # Görselleştirme
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Orijinal Görüntü")
        plt.imshow(test_image)

        plt.subplot(1, 2, 2)
        plt.title("Tahmin Edilen Mask")
        plt.imshow(output, cmap="gray")
        plt.show()
