import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- CẤU HÌNH ĐÁNH GIÁ ---
CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "BATCH_SIZE": 64,           
    "EPOCHS": 20,               # 20 Epoch là đủ để đánh giá Linear Probing
    "LEARNING_RATE": 1e-3,      
    "IMG_SIZE": 128,            # Resize lên 128 để khớp với Pretrain
    "NUM_CLASSES": 100,         # CIFAR-100
    # Đường dẫn tới file backbone tốt nhất bạn vừa train xong
    "PRETRAIN_PATH": "./models/backbone_best.pth", 
    "FREEZE_BACKBONE": False     # True: Chỉ train lớp cuối (Đánh giá chất lượng Feature)
                                # False: Train toàn bộ (Fine-tune để lấy Top Accuracy)
}

print(f"Running CIFAR-100 Evaluation on: {CONFIG['DEVICE']}")
print(f"Mode: {'Linear Probing (Freeze Backbone)' if CONFIG['FREEZE_BACKBONE'] else 'Fine-tuning (Unfreeze All)'}")

# ==============================================================================
# 1. ĐỊNH NGHĨA KIẾN TRÚC BACKBONE (Phải khớp 100% với lúc Pretrain)
# ==============================================================================
class PixelPositionEncoding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        N, C, H, W = x.shape
        y_coords = torch.linspace(0, 1, H, device=x.device)
        x_coords = torch.linspace(0, 1, W, device=x.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        xx = xx.unsqueeze(0).unsqueeze(0).repeat(N, 1, 1, 1)
        yy = yy.unsqueeze(0).unsqueeze(0).repeat(N, 1, 1, 1)
        return torch.cat([x, xx, yy], dim=1)

class MultiKernelSpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, dropout=0.1):
        super().__init__()
        self.configs = [(1, 1), (3, 1), (3, 2), (3, 5), (3, 6)]
        
        if in_channels < 16:
            self.inter_channels = in_channels
            self.reduce_conv = nn.Sequential(nn.Conv2d(in_channels, self.inter_channels, 1, bias=False), nn.BatchNorm2d(self.inter_channels), nn.ReLU(True))
        else:
            self.inter_channels = max(8, out_channels // reduction)
            self.reduce_conv = nn.Sequential(nn.Conv2d(in_channels, self.inter_channels, 1, bias=False), nn.BatchNorm2d(self.inter_channels), nn.ReLU(True))
        
        self.feat_convs = nn.ModuleList()
        self.conf_convs = nn.ModuleList()
        for k, d in self.configs:
            pad = ((k - 1) * d) // 2
            self.feat_convs.append(nn.Sequential(nn.Conv2d(self.inter_channels, self.inter_channels, k, padding=pad, dilation=d, bias=False), nn.BatchNorm2d(self.inter_channels), nn.Sigmoid()))
            self.conf_convs.append(nn.Sequential(nn.Conv2d(self.inter_channels, 1, k, padding=pad, dilation=d, bias=False), nn.BatchNorm2d(1)))
        
        self.fusion = nn.Sequential(nn.Conv2d(self.inter_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True), nn.Dropout2d(dropout))

    def forward(self, x):
        x_reduced = self.reduce_conv(x) 
        feats, confs = [], []
        for i in range(len(self.configs)):
            feats.append(self.feat_convs[i](x_reduced))
            confs.append(self.conf_convs[i](x_reduced))
        stack_confs = torch.cat(confs, dim=1) 
        weights = F.softmax(stack_confs, dim=1)
        split_weights = torch.chunk(weights, len(self.configs), dim=1)
        weighted_sum = sum(feats[i] * split_weights[i] for i in range(len(self.configs)))
        return self.fusion(weighted_sum)

class HybridEncoder(nn.Module):
    """
    Backbone chứa các trọng số đã học từ Autoencoder.
    """
    def __init__(self):
        super().__init__()
        self.pos_enc = PixelPositionEncoding()
        
        # Các layer này tên phải khớp với file .pth đã lưu
        self.mk1 = MultiKernelSpatialAttention(5, 64)   
        self.ds1 = nn.MaxPool2d(2, 2)
        self.mk2 = MultiKernelSpatialAttention(64, 128) 
        self.ds2 = nn.MaxPool2d(2, 2)
        self.mk3 = MultiKernelSpatialAttention(128, 256) 
        self.ds3 = nn.MaxPool2d(2, 2)
        self.mk_bot = MultiKernelSpatialAttention(256, 512) 
        
    def forward(self, x):
        x = self.pos_enc(x)
        x1 = self.mk1(x)
        x1_d = self.ds1(x1)
        x2 = self.mk2(x1_d)
        x2_d = self.ds2(x2)
        x3 = self.mk3(x2_d)
        x3_d = self.ds3(x3)
        bot = self.mk_bot(x3_d)
        return bot 

# ==============================================================================
# 2. MÔ HÌNH PHÂN LOẠI (Classifier)
# ==============================================================================
class CIFAR100Classifier(nn.Module):
    def __init__(self, num_classes=100, pretrain_path=None):
        super().__init__()
        # 1. Backbone
        self.backbone = HybridEncoder()
        
        # 2. Load Pretrain
        if pretrain_path and os.path.exists(pretrain_path):
            print(f">>> Loading backbone weights from: {pretrain_path}")
            try:
                state_dict = torch.load(pretrain_path, map_location='cpu')
                # Load với strict=False để an toàn, nhưng vì ta định nghĩa giống hệt nên sẽ khớp
                msg = self.backbone.load_state_dict(state_dict, strict=True)
                print(f"    Load success: {msg}")
            except Exception as e:
                print(f"!!! Error loading weights: {e}")
        else:
            print("!!! WARNING: Pretrain path not found. Training from scratch!")

        # 3. Head (Classification)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.backbone(x) # [B, 512, H, W]
        x = self.pool(features)     # [B, 512, 1, 1]
        x = self.flatten(x)         # [B, 512]
        logits = self.fc(x)         # [B, 100]
        return logits

# ==============================================================================
# 3. CHUẨN BỊ DỮ LIỆU
# ==============================================================================
def get_dataloaders():
    # Mean/Std của CIFAR-100
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    
    transform_train = transforms.Compose([
        transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # Autoencoder pretrain không dùng normalize, nhưng Classifier thường cần.
        # Tuy nhiên để tương thích tốt nhất, ta thử KHÔNG normalize trước, hoặc normalize nhẹ.
        # Ở đây dùng normalize chuẩn của CIFAR để hội tụ lớp Linear nhanh hơn.
        transforms.Normalize(*stats), 
    ])

    transform_test = transforms.Compose([
        transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=CONFIG["BATCH_SIZE"],
                             shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=CONFIG["BATCH_SIZE"],
                            shuffle=False, num_workers=2, pin_memory=True)
    
    return trainloader, testloader

# ==============================================================================
# 4. TRAINING LOOP
# ==============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")
        
    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

def main():
    trainloader, testloader = get_dataloaders()
    
    model = CIFAR100Classifier(
        num_classes=CONFIG["NUM_CLASSES"], 
        pretrain_path=CONFIG["PRETRAIN_PATH"]
    ).to(CONFIG["DEVICE"])
    
    # --- XỬ LÝ FREEZE/UNFREEZE ---
    if CONFIG["FREEZE_BACKBONE"]:
        print(">>> Freezing Backbone... Only training the Head.")
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        print(">>> Unfrozen Backbone... Training EVERYTHING.")
            
    # Chỉ đưa những tham số requires_grad=True vào optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(trainable_params, lr=CONFIG["LEARNING_RATE"], weight_decay=1e-4)
    
    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG["LEARNING_RATE"], 
                                              steps_per_epoch=len(trainloader), 
                                              epochs=CONFIG["EPOCHS"])

    print(f"\n>>> Start Training CIFAR-100 for {CONFIG['EPOCHS']} epochs...")
    best_acc = 0.0
    
    for epoch in range(CONFIG["EPOCHS"]):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, CONFIG["DEVICE"])
        val_loss, val_acc = evaluate(model, testloader, criterion, CONFIG["DEVICE"])
        
        # Step scheduler
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{CONFIG['EPOCHS']}] "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Lưu model tốt nhất
            torch.save(model.state_dict(), "cifar100_best_result.pth")
            print(f"--> New Best Acc: {best_acc:.2f}% (Saved)")

    print(f"\n>>> DONE! Best CIFAR-100 Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()