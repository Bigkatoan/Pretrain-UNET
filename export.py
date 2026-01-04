import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json

# --- CẤU HÌNH EXPORT ---
EXPORT_CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_CLASSES": 81,          # Số class lúc train
    "IMG_SIZE": 128,            # Kích thước đầu vào dummy
    "CHECKPOINT_PATH": "./artifacts_hybrid_model/hybrid_model_ep50.pth", # Đường dẫn checkpoint tốt nhất
    "OUTPUT_DIR": "./pretrained_models"
}

os.makedirs(EXPORT_CONFIG["OUTPUT_DIR"], exist_ok=True)
print(f"Running Export on: {EXPORT_CONFIG['DEVICE']}")

# ==============================================================================
# 1. ĐỊNH NGHĨA CÁC KHỐI CƠ BẢN (Cần thiết để tái tạo model)
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
        
        # Logic giảm kênh (Bottleneck)
        if in_channels < 16:
            self.inter_channels = in_channels
            self.reduce_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.inter_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.inter_channels = max(8, out_channels // reduction)
            self.reduce_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.inter_channels),
                nn.ReLU(inplace=True)
            )

        self.feat_convs = nn.ModuleList()
        self.conf_convs = nn.ModuleList()
        
        for k, d in self.configs:
            pad = ((k - 1) * d) // 2
            self.feat_convs.append(nn.Sequential(
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=k, padding=pad, dilation=d, bias=False),
                nn.BatchNorm2d(self.inter_channels),
                nn.Sigmoid()
            ))
            self.conf_convs.append(nn.Sequential(
                nn.Conv2d(self.inter_channels, 1, kernel_size=k, padding=pad, dilation=d, bias=False),
                nn.BatchNorm2d(1)
            ))
            
        self.fusion = nn.Sequential(
            nn.Conv2d(self.inter_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) 
        )

    def forward(self, x):
        x_reduced = self.reduce_conv(x) 
        feats = []
        confs = []
        for i in range(len(self.configs)):
            feats.append(self.feat_convs[i](x_reduced))
            confs.append(self.conf_convs[i](x_reduced))
        stack_confs = torch.cat(confs, dim=1) 
        weights = F.softmax(stack_confs, dim=1)
        split_weights = torch.chunk(weights, len(self.configs), dim=1)
        weighted_sum = 0
        for i in range(len(self.configs)):
            weighted_sum += feats[i] * split_weights[i]
        return self.fusion(weighted_sum)

# ==============================================================================
# 2. MODEL 1: FULL SEGMENTATION PRETRAIN (Kiến trúc gốc)
# ==============================================================================
class HybridMultiKernelNetwork(nn.Module):
    def __init__(self, num_classes=81):
        super().__init__()
        self.pos_enc = PixelPositionEncoding()
        
        # Encoder
        self.mk1 = MultiKernelSpatialAttention(5, 64)   
        self.ds1 = nn.MaxPool2d(2, 2)
        self.mk2 = MultiKernelSpatialAttention(64, 128) 
        self.ds2 = nn.MaxPool2d(2, 2)
        self.mk3 = MultiKernelSpatialAttention(128, 256) 
        self.ds3 = nn.MaxPool2d(2, 2)
        self.mk_bot = MultiKernelSpatialAttention(256, 512) 
        
        # Decoder
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 2, 2, bias=False), nn.BatchNorm2d(256), nn.ReLU(True))
        self.mk_dec1 = MultiKernelSpatialAttention(512, 256) 
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, 2, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.mk_dec2 = MultiKernelSpatialAttention(256, 128)
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, 2, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.mk_dec3 = MultiKernelSpatialAttention(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.pos_enc(x)
        # Encoder
        x1 = self.mk1(x)
        x1_d = self.ds1(x1)
        x2 = self.mk2(x1_d)
        x2_d = self.ds2(x2)
        x3 = self.mk3(x2_d)
        x3_d = self.ds3(x3)
        bot = self.mk_bot(x3_d)
        # Decoder
        u1 = self.up1(bot)
        if u1.size() != x3.size(): u1 = F.interpolate(u1, size=x3.shape[2:])
        c1 = torch.cat([u1, x3], dim=1)
        d1 = self.mk_dec1(c1)
        u2 = self.up2(d1)
        if u2.size() != x2.size(): u2 = F.interpolate(u2, size=x2.shape[2:])
        c2 = torch.cat([u2, x2], dim=1)
        d2 = self.mk_dec2(c2)
        u3 = self.up3(d2)
        if u3.size() != x1.size(): u3 = F.interpolate(u3, size=x1.shape[2:])
        c3 = torch.cat([u3, x1], dim=1)
        d3 = self.mk_dec3(c3)
        out = self.final_conv(d3)
        return out

# ==============================================================================
# 3. MODEL 2: ENCODER (EMBEDDING) ONLY
# ==============================================================================
class HybridEncoder(nn.Module):
    """
    Chỉ trích xuất phần Backbone để làm Embedding Model.
    Output: Feature Map [B, 512, H/8, W/8]
    """
    def __init__(self):
        super().__init__()
        self.pos_enc = PixelPositionEncoding()
        
        # Tên biến phải GIỐNG HỆT model gốc để load_state_dict tự khớp
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
        return bot # Embedding (Spatial Features)

# ==============================================================================
# 4. EXPORT FUNCTIONS
# ==============================================================================
def export_models():
    # 1. Load Full Checkpoint
    print(f">>> Loading Checkpoint: {EXPORT_CONFIG['CHECKPOINT_PATH']}")
    if not os.path.exists(EXPORT_CONFIG['CHECKPOINT_PATH']):
        print("!!! Checkpoint not found. Please train first.")
        return

    full_model = HybridMultiKernelNetwork(num_classes=EXPORT_CONFIG["NUM_CLASSES"]).to(EXPORT_CONFIG["DEVICE"])
    checkpoint = torch.load(EXPORT_CONFIG["CHECKPOINT_PATH"], map_location=EXPORT_CONFIG["DEVICE"])
    full_model.load_state_dict(checkpoint)
    full_model.eval()
    print(">>> Full Model Loaded.")

    # 2. Export Full Segmentation Model
    print("\n--- Exporting 1: Segmentation Pretrain ---")
    seg_path_pth = os.path.join(EXPORT_CONFIG["OUTPUT_DIR"], "hybrid_segmentation_full.pth")
    seg_path_pt  = os.path.join(EXPORT_CONFIG["OUTPUT_DIR"], "hybrid_segmentation_portable.pt")
    
    # Save Weights (.pth)
    torch.save(full_model.state_dict(), seg_path_pth)
    print(f"Saved Weights: {seg_path_pth}")
    
    # Save TorchScript (.pt)
    dummy_input = torch.randn(1, 3, EXPORT_CONFIG["IMG_SIZE"], EXPORT_CONFIG["IMG_SIZE"]).to(EXPORT_CONFIG["DEVICE"])
    traced_seg = torch.jit.trace(full_model, dummy_input)
    traced_seg.save(seg_path_pt)
    print(f"Saved Portable: {seg_path_pt}")

    # 3. Export Encoder (Embedding) Model
    print("\n--- Exporting 2: Encoder (Embedding) Pretrain ---")
    encoder_model = HybridEncoder().to(EXPORT_CONFIG["DEVICE"])
    
    # Load weights từ Full Model sang Encoder
    # strict=False là CHÌA KHOÁ: nó sẽ load các layer khớp tên (mk1, mk2...) và bỏ qua các layer thừa (decoder)
    encoder_model.load_state_dict(full_model.state_dict(), strict=False)
    encoder_model.eval()
    
    enc_path_pth = os.path.join(EXPORT_CONFIG["OUTPUT_DIR"], "hybrid_encoder_embedding.pth")
    enc_path_pt  = os.path.join(EXPORT_CONFIG["OUTPUT_DIR"], "hybrid_encoder_portable.pt")
    
    # Save Weights (.pth)
    torch.save(encoder_model.state_dict(), enc_path_pth)
    print(f"Saved Weights: {enc_path_pth}")
    
    # Save TorchScript (.pt)
    traced_enc = torch.jit.trace(encoder_model, dummy_input)
    traced_enc.save(enc_path_pt)
    print(f"Saved Portable: {enc_path_pt}")
    
    # Test Output Shape
    with torch.no_grad():
        emb = encoder_model(dummy_input)
    print(f"\n>>> Verify Embedding Shape: {emb.shape}")
    print("    (Expect: [1, 512, H/8, W/8])")
    print("\n>>> ALL DONE! Models are ready in:", EXPORT_CONFIG["OUTPUT_DIR"])

def verify_exported_models():
    """
    Hàm demo cách load và sử dụng các model đã export (.pt và .pth).
    Giúp kiểm tra tính toàn vẹn của model sau khi export.
    """
    print("\n" + "="*50)
    print("DEMO: HƯỚNG DẪN LOAD VÀ SỬ DỤNG MODEL EXPORT")
    print("="*50)
    
    device = EXPORT_CONFIG["DEVICE"]
    # Input giả lập để test inference
    dummy_input = torch.randn(1, 3, EXPORT_CONFIG["IMG_SIZE"], EXPORT_CONFIG["IMG_SIZE"]).to(device)
    
    # --- CÁCH 1: LOAD MODEL PORTABLE (.pt - TorchScript) ---
    # Ưu điểm: Không cần file code python chứa class definition, chạy được trên C++
    print("\n[CÁCH 1] Load TorchScript Model (.pt) - KHÔNG CẦN CLASS DEF")
    
    # 1.1 Load Segmentation Model
    seg_pt_path = os.path.join(EXPORT_CONFIG["OUTPUT_DIR"], "hybrid_segmentation_portable.pt")
    if os.path.exists(seg_pt_path):
        print(f"  -> Loading: {seg_pt_path}")
        model_seg = torch.jit.load(seg_pt_path, map_location=device)
        model_seg.eval()
        with torch.no_grad():
            output = model_seg(dummy_input)
        print(f"     Output Shape: {output.shape} (Segmentation Mask)")

    # 1.2 Load Encoder Model
    enc_pt_path = os.path.join(EXPORT_CONFIG["OUTPUT_DIR"], "hybrid_encoder_portable.pt")
    if os.path.exists(enc_pt_path):
        print(f"  -> Loading: {enc_pt_path}")
        model_enc = torch.jit.load(enc_pt_path, map_location=device)
        model_enc.eval()
        with torch.no_grad():
            emb = model_enc(dummy_input)
        print(f"     Output Shape: {emb.shape} (Feature Embedding)")

    # --- CÁCH 2: LOAD MODEL WEIGHTS (.pth - PyTorch Standard) ---
    # Ưu điểm: Dễ dàng fine-tune tiếp, nhưng CẦN file code chứa class definition
    print("\n[CÁCH 2] Load Weights (.pth) - CẦN CLASS DEFINITION")
    
    # 2.1 Load Encoder Weights (Ví dụ để fine-tune cho bài toán khác)
    enc_pth_path = os.path.join(EXPORT_CONFIG["OUTPUT_DIR"], "hybrid_encoder_embedding.pth")
    if os.path.exists(enc_pth_path):
        print(f"  -> Loading Weights: {enc_pth_path}")
        # Bước 1: Khởi tạo kiến trúc (phải có class HybridEncoder ở trên)
        new_encoder = HybridEncoder().to(device)
        # Bước 2: Load weights
        new_encoder.load_state_dict(torch.load(enc_pth_path, map_location=device))
        new_encoder.eval()
        
        with torch.no_grad():
            emb_pth = new_encoder(dummy_input)
        print(f"     Output Shape: {emb_pth.shape} (Feature Embedding)")
        print("     -> Model sẵn sàng để Fine-tune!")

    print("\n>>> DEMO HOÀN TẤT. CÁC MODEL HOẠT ĐỘNG TỐT.")

if __name__ == "__main__":
    export_models()
    verify_exported_models()