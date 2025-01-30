import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import kagglehub


# Download dataset
path = kagglehub.dataset_download("manideep1108/tusimple")
print("Path to dataset files:", path)



# Kayıp Fonksiyonu Sınıfı
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weights={'focal': 0.75, 'dice': 0.25}):
       
        super().__init__()
        self.alpha = alpha  
        self.gamma = gamma  
        self.weights = weights  

    def forward(self, inputs, targets):
      
        # Binary Cross-Entropy kaybını hesapla
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # pt: Tahminin doğruluk seviyesi
        pt = torch.exp(-bce_loss)
        
        # Focal Loss hesaplama
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        focal_loss = focal_loss.mean()  # Ortalama focal loss
        
        # Dice Loss hesaplama
        dice_loss = 1 - dice_coefficient(inputs, targets)
        
        # Focal ve Dice Loss'u ağırlıklandırarak birleştir
        return self.weights['focal'] * focal_loss + self.weights['dice'] * dice_loss


# Dice Coefficient Fonksiyonu
def dice_coefficient(pred, target):
    """Dice katsayısını hesaplar
    """
    smooth = 1e-5 
    intersection = (pred * target).sum()  
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


# Intersection over Union (IoU) Fonksiyonu
def iou_score(pred, target):
    """Intersection over Union (IoU) skorunu hesaplar."""
    smooth = 1e-5 
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection  
    return (intersection + smooth) / (union + smooth)


# Veri Artırma Sınıfı
class LaneAugmentation:
    def __init__(self, p=0.5):
        """Segmentasyon görevlerinde veri setini zenginleştirmek için çeşitli dönüşümler uygular.
        """
        self.p = p 

    def __call__(self, image, mask):
        
        # Rastgele rotasyon
        if random.random() < self.p:
            angle = random.uniform(-10, 10) 
            image = TF.rotate(image, angle) 
            mask = TF.rotate(mask.unsqueeze(0), angle).squeeze(0)  
        
        # Rastgele perspektif dönüşümü
        if random.random() < self.p:
            width, height = image.shape[-2:] 
            startpoints = [[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]]  
            endpoints = [[random.randint(-20, 20), random.randint(-20, 20)],
                         [width-1 + random.randint(-20, 20), random.randint(-20, 20)],
                         [random.randint(-20, 20), height-1 + random.randint(-20, 20)],
                         [width-1 + random.randint(-20, 20), height-1 + random.randint(-20, 20)]]
            image = TF.perspective(image, startpoints, endpoints) 
            mask = TF.perspective(mask.unsqueeze(0), startpoints, endpoints).squeeze(0) 
        
   
        if random.random() < self.p:
            brightness_factor = random.uniform(0.8, 1.2) 
            contrast_factor = random.uniform(0.8, 1.2)  
            image = TF.adjust_brightness(image, brightness_factor)  
            image = TF.adjust_contrast(image, contrast_factor)  
        
        return image, mask  



# Veri seti sınıfı
class TuSimpleDataset(Dataset):
    def __init__(self, data_dir, label_files, transform=None, augment=True):
       
        self.data_dir = data_dir  # Veri dizini
        self.base_transform = transform  # Görüntülere uygulanacak dönüşümler
        self.augment = augment  # Veri artırma seçeneği
        self.lane_augment = LaneAugmentation() if augment else None  # Veri artırma işlemleri
        self.data = []  # Etiketli veri listesi

        # Etiket dosyalarını yükle
        for label_file in label_files:
            if not os.path.exists(label_file):
                raise FileNotFoundError(f"Label file not found: {label_file}")  
            with open(label_file, 'r') as f:
                self.data.extend([json.loads(line) for line in f])  # JSON formatındaki her satırı yükle

        print(f"Loaded {len(self.data)} samples from {len(label_files)} files")  

        # Pozitif sınıf (şerit) ağırlığını hesapla
        total_pixels = 0
        lane_pixels = 0
        for item in self.data[:100]:  # Sadece ilk 100 örneği kullanarak ağırlığı hesapla
            mask = self._create_mask(item['lanes'], item['h_samples'], (224, 224))
            total_pixels += mask.numel()  # Toplam piksel sayısı
            lane_pixels += mask.sum().item()  # Şerite ait toplam piksel sayısı

       
        self.pos_weight = (total_pixels - lane_pixels) / (lane_pixels + 1e-6)
        print(f"Positive class weight: {self.pos_weight:.2f}")

    def __len__(self):
       
        return len(self.data)

    def _create_mask(self, lanes, h_samples, size):
       
        mask = torch.zeros(size, dtype=torch.float32)  # Boyutları (height, width) olan sıfırlarla dolu tensor
        for lane in lanes:
            for x, y in zip(lane, h_samples):  # Şeritteki her (x, y) koordinatını kontrol et
                if x != -1:  # Geçerli bir x koordinatı varsa
                    # x ve y koordinatlarını maskenin boyutuna ölçeklendir
                    x_rescaled = int(x * size[1] / 1280)
                    y_rescaled = int(y * size[0] / 720)
                    if 0 <= x_rescaled < size[1] and 0 <= y_rescaled < size[0]:  # Koordinatların sınır içinde olduğundan emin ol
                        # Gaussian kernel ile yumuşak bir maske oluştur
                        sigma = 1.0  # Gaussian için standart sapma
                        kernel_size = 5  # Kernel boyutu
                        for i in range(-kernel_size // 2, kernel_size // 2 + 1):
                            for j in range(-kernel_size // 2, kernel_size // 2 + 1):
                                if (0 <= x_rescaled + i < size[1] and 
                                    0 <= y_rescaled + j < size[0]):
                                    # Gaussian ağırlık değerini hesapla
                                    gaussian_value = torch.tensor(-(i**2 + j**2) / (2 * sigma**2))
                                    mask[y_rescaled + j, x_rescaled + i] = torch.exp(gaussian_value)
        return mask

    def __getitem__(self, idx):
      
        item = self.data[idx]  # İlgili etiketi al
        img_path = os.path.join(self.data_dir, item['raw_file'])  # Görüntü dosyasının yolu

        if not os.path.exists(img_path):  # Dosya mevcut değilse hata fırlat
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # Görüntüyü yükle ve RGB formatına dönüştür
        image = Image.open(img_path).convert("RGB")
        
        # Şerit bilgilerini kullanarak maske oluştur
        mask = self._create_mask(item['lanes'], item['h_samples'], (224, 224))

        # Görüntüye dönüşüm uygula
        if self.base_transform:
            image = self.base_transform(image)

        # Veri artırma uygula
        if self.augment and self.lane_augment:
            image, mask = self.lane_augment(image, mask)

        return image, mask  # Görüntü ve maske döndürülür



# UNet tarzı bir çift konvolüsyon bloğunu tanımlayan sınıf
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
     
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # İlk konvolüsyon
            nn.BatchNorm2d(out_channels),                                    # Batch normalization
            nn.ReLU(inplace=True),                                           # Aktivasyon (ReLU)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # İkinci konvolüsyon
            nn.BatchNorm2d(out_channels),                                    # Batch normalization
            nn.ReLU(inplace=True)                                            # Aktivasyon (ReLU)
        )

    def forward(self, x):
       return self.double_conv(x)


# SCSE (Spatial and Channel Squeeze & Excitation) modülü
class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
       
        super().__init__()
        # Kanal Squeeze ve Excitation (cSE) Bloğu
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                      
            nn.Conv2d(in_channels, in_channels // reduction, 1), 
            nn.ReLU(inplace=True),                       
            nn.Conv2d(in_channels // reduction, in_channels, 1),  
            nn.Sigmoid()                                  
        )
        # Uzamsal Squeeze ve Excitation (sSE) Bloğu
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),                  
            nn.Sigmoid()                               
        )

    def forward(self, x):
       
       
        return x * self.cSE(x) + x * self.sSE(x)



# ResNet tabanlı UNet mimarisi tanımlanıyor
class ResNetUNet(nn.Module):
    def __init__(self, n_classes=1):
        """ ResNet-50 tabanlı UNet modelini tanımlar"""
        super().__init__()
        
        # ResNet-50'yi önceden eğitilmiş ağırlıklarla yükle
        self.base_model = models.resnet50(pretrained=True)

        # Encoder1: ResNet'in ilk konvolüsyon katmanlarını al
        self.encoder1 = nn.Sequential(
            self.base_model.conv1, 
            self.base_model.bn1,    
            self.base_model.relu    
        )
        self.pool = self.base_model.maxpool  

        # ResNet katmanları Encoder olarak tanımlanıyor
        self.encoder2 = self.base_model.layer1 
        self.encoder3 = self.base_model.layer2  
        self.encoder4 = self.base_model.layer3 

        # Decoder için upsampling katmanları ve DoubleConv modülleri
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) 
        self.decoder4 = DoubleConv(1024, 512)  

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  
        self.decoder3 = DoubleConv(512, 256)  

        self.upconv2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)  
        self.decoder2 = DoubleConv(128, 64)  

        # Çözünürlüğü orijinal boyuta döndürmek için son upsampling ve konvolüsyon katmanları
        self.upconv_final = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),                         
            nn.ReLU(inplace=True),                      
            nn.Conv2d(32, n_classes, kernel_size=1),    
            nn.Sigmoid()                                 
        )

        self.scse = SCSEModule(64)

    def forward(self, x):
        """ İleri yönlü veri akışı: Girişten çıkışa veri aktarımı. """
        # Encoder'dan özellik çıkarımı
        enc1 = self.encoder1(x)        
        enc1 = self.scse(enc1)         
        enc2 = self.encoder2(self.pool(enc1))  
        enc3 = self.encoder3(enc2)   
        enc4 = self.encoder4(enc3)    

        # Decoder aşaması: Özellik haritasını yeniden orijinal boyuta döndür
        dec4 = self.upconv4(enc4)                      
        dec4 = torch.cat((dec4, enc3), dim=1)           
        dec4 = self.decoder4(dec4)                   

        dec3 = self.upconv3(dec4)                     
        dec3 = torch.cat((dec3, enc2), dim=1)          
        dec3 = self.decoder3(dec3)                    

        dec2 = self.upconv2(dec3)                       
        dec2 = torch.cat((dec2, enc1), dim=1)           
        dec2 = self.decoder2(dec2)                     

       
        x = self.upconv_final(dec2)                   
        x = self.final_conv(x)                         

        return x  

    


# Eğitim fonksiyonu
def train_model(model, train_loader, val_loader, num_epochs=50):
    """Modeli eğitmek ve doğrulamak için bir eğitim döngüsü uygular."""
    # Cihaz (CPU veya GPU) ayarlanır
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Modeli cihaza aktar
    
    # Kayıp fonksiyonu olarak FocalLoss tanımlanır
    criterion = FocalLoss(alpha=1, gamma=2)
    
    # AdamW optimizasyon algoritması tanımlanır
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Öğrenme oranı planlayıcısı tanımlanır (OneCycleLR)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # En iyi doğrulama kaybını takip etmek için değişkenler
    best_val_loss = float('inf')
    patience = 10 
    patience_counter = 0
    
    # Eğitim döngüsü
    for epoch in range(num_epochs):
        # **Eğitim Aşaması**
        model.train()  
        train_loss = 0  

        # Eğitim veri kümesi üzerinden iterasyon
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device) 
            
            optimizer.zero_grad()  
            outputs = model(images) 
            loss = criterion(outputs, masks.unsqueeze(1))  
            loss.backward()  
            
            # Gradients değerlerini sınırlayarak patlamayı önle
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()  
            scheduler.step()  
            
            train_loss += loss.item()  
        
        # **Doğrulama Aşaması**
        model.eval()  
        val_loss = 0
        val_dice = 0
        val_iou = 0

        # Doğrulama veri kümesi üzerinden iterasyon (gradients yok)
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks.unsqueeze(1)).item()
                val_dice += dice_coefficient(outputs, masks.unsqueeze(1)).item()
                val_iou += iou_score(outputs, masks.unsqueeze(1)).item()
        
        # Ortalama kayıpları hesapla
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        
        # Eğitim ve doğrulama sonuçlarını yazdır
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Dice: {val_dice:.4f}')
        print(f'Validation IoU: {val_iou:.4f}')
        
        # En iyi modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model.pth')
        else:
            patience_counter += 1  # Erken durdurma için sayaç artır
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break


# Ana çalışma fonksiyonu
if __name__ == "__main__":
    # Eğitim veri yollarını ayarla
    train_data_dir = "/root/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/train_set"
    train_label_files = [
        os.path.join(train_data_dir, "label_data_0313.json"),
        os.path.join(train_data_dir, "label_data_0531.json")
    ]
    val_label_file = os.path.join(train_data_dir, "label_data_0601.json")

    # Görüntü dönüşümleri tanımla
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Görüntü boyutlarını 224x224 olarak yeniden boyutlandır
        transforms.ToTensor(),         # Görüntüyü PyTorch tensorüne dönüştür
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize et
    ])

    # Eğitim ve doğrulama veri kümelerini oluştur
    train_dataset = TuSimpleDataset(train_data_dir, train_label_files, transform=transform, augment=True)
    val_dataset = TuSimpleDataset(train_data_dir, [val_label_file], transform=transform, augment=False)

    # Veri yükleyicilerini oluştur
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Modeli başlat
    model = ResNetUNet(n_classes=1)

    # Modeli eğit
    num_epochs = 50
    train_model(model, train_loader, val_loader, num_epochs=num_epochs)

    print("Training complete. Best model saved as 'best_model.pth'.")



# Test için gerekli modüller
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm
#######3
# model = ResNetUNet(n_classes=1)
####
#######################3

#   # Define transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#####################
# Test veri seti ve yükleyici
test_data_dir = "/root/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/test_set"
test_label_file = os.path.join(test_data_dir, "test_tasks_0627.json")
test_dataset = TuSimpleDataset(test_data_dir, [test_label_file], transform=transform, augment=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# Modeli yükleme ve değerlendirme
def test_model(model_path, test_loader, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Kaydedilmiş modeli yükle
    print("Loading model...")
    model = ResNetUNet()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Performans ölçütleri
    dice_scores = []
    iou_scores = []
    all_predictions = []
    all_targets = []

    print("Evaluating model...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)

            # Model tahmini
            outputs = model(images)
            predictions = (outputs > threshold).float()

            # Dice ve IoU skorlarını hesapla
            dice = dice_coefficient(predictions, masks.unsqueeze(1)).item()
            iou = iou_score(predictions, masks.unsqueeze(1)).item()
            dice_scores.append(dice)
            iou_scores.append(iou)

            # PR eğrisi için tahminleri ve gerçek değerleri topla
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

    # Precision-Recall eğrisi ve AP skoru
    precision, recall, _ = precision_recall_curve(all_targets, all_predictions)
    ap_score = average_precision_score(all_targets, all_predictions)

    # PR eğrisini çiz
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AP Score: {ap_score:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.savefig("pr_curve_test.png")
    plt.close()

    # Metrik sonuçları
    metrics = {
        "Dice Score (avg)": np.mean(dice_scores),
        "IoU Score (avg)": np.mean(iou_scores),
        "Average Precision": ap_score,
    }

    return metrics

# Test çalıştırma ve sonuçları yazdırma
if __name__ == "__main__":
    model_path = "best_model.pth"
    metrics = test_model(model_path, test_loader)

    print("\nTest Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("Precision-Recall curve saved as 'pr_curve_test.png'")



# Test için model yükleme
model = ResNetUNet(n_classes=1)
checkpoint = torch.load('best_model.pth')

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    Model performansını test seti üzerinde değerlendirir ve detaylı metrikler hesaplar.
    """
    model.eval()

    # Metrik değerlerini tutacak listeler
    all_predictions = []
    all_targets = []
    dice_scores = []
    iou_scores = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)

            # Model tahminleri
            outputs = model(images)
            predictions = (outputs > threshold).float()

            # Batch için metrikleri hesapla
            dice = dice_coefficient(predictions, masks.unsqueeze(1)).item()
            iou = iou_score(predictions, masks.unsqueeze(1)).item()
            acc, prec, rec, f1 = calculate_metrics(predictions.cpu().numpy(),
                                                 masks.unsqueeze(1).cpu().numpy())

            # Metrikleri listelere ekle
            dice_scores.append(dice)
            iou_scores.append(iou)
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)

            # PR eğrisi için tüm tahminleri ve gerçek değerleri sakla
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(masks.unsqueeze(1).cpu().numpy().flatten())

    # Ortalama metrikleri hesapla
    metrics = {
        'Dice Score': np.mean(dice_scores),
        'IoU Score': np.mean(iou_scores),
        'Accuracy': np.mean(accuracies),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'F1 Score': np.mean(f1_scores)
    }

    # PR eğrisi ve AP skorunu hesapla
    precision, recall, _ = precision_recall_curve(all_targets, all_predictions)
    ap_score = average_precision_score(all_targets, all_predictions)
    metrics['Average Precision'] = ap_score

    # PR eğrisini çiz
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, 'b-', label=f'PR Curve (AP = {ap_score:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('pr_curve.png')
    plt.close()

    return metrics

def calculate_metrics(y_pred, y_true):
    """
    Temel sınıflandırma metriklerini hesaplar
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    # True Positive, False Positive, True Negative, False Negative
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Metrikleri hesapla
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    return accuracy, precision, recall, f1

def visualize_predictions(model, test_loader, device, num_samples=5):
    """
    Test setinden rastgele örnekler seçer ve tahminleri görselleştirir
    """
    model.eval()

    # Rastgele örnekler için
    data_iter = iter(test_loader)
    images, masks = next(data_iter)

    with torch.no_grad():
        images = images.to(device)
        predictions = model(images)
        predictions = (predictions > 0.5).float()

    # Görselleştirme
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    for idx in range(min(num_samples, len(images))):
        # Orijinal görüntü
        axes[idx, 0].imshow(images[idx].cpu().permute(1, 2, 0))
        axes[idx, 0].set_title('Input Image')
        axes[idx, 0].axis('off')

        # Gerçek mask
        axes[idx, 1].imshow(masks[idx].cpu(), cmap='gray')
        axes[idx, 1].set_title('Ground Truth')
        axes[idx, 1].axis('off')

        # Tahmin
        axes[idx, 2].imshow(predictions[idx, 0].cpu(), cmap='gray')
        axes[idx, 2].set_title('Prediction')
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

# Test ve görselleştirme için kullanım örneği
def run_evaluation(model_path, test_loader):
    """
    Kaydedilmiş modeli yükler ve tüm değerlendirme işlemlerini gerçekleştirir
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Modeli yükle
    model = ResNetUNet()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Metrikleri hesapla
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device)

    # Sonuçları yazdır
    print("\nTest Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Örnek tahminleri görselleştir
    print("\nGenerating visualization...")
    visualize_predictions(model, test_loader, device)
    print("Visualizations saved as 'predictions.png'")
    print("PR curve saved as 'pr_curve.png'")

    return metrics

# Test Veri Seti
import os
test_data_dir = "/root/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/test_set"
test_label_file = os.path.join(test_data_dir, "test_tasks_0627.json")
test_dataset = TuSimpleDataset(test_data_dir, [test_label_file], transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# Model değerlendirmesi
metrics = run_evaluation('best_model.pth', test_loader)

