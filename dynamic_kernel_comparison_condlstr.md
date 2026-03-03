# DynamicKernelHead vs CondLSTR — Karşılaştırma

## CondLSTR Implementasyonu (/home/alki/projects/CondLSTR)

CondLSTR'de **`DynamicMaskHead`** kullanılıyor, bu farklı bir yaklaşım:

### CondLSTR DynamicMaskHead Yapısı

```python
# CondLSTR: cond_lstr_2d.py satır 14-93
class DynamicMaskHead(nn.Module):
    def forward(self, x, mask_head_params, is_mask=True):
        # x: (N, C, H, W) feature map
        # mask_head_params: (N, L, num_params) decoder'dan gelen
        
        # 1. Parametreleri parse et → weights, biases
        weights, biases = self.parse_dynamic_params(mask_head_params)
        
        # 2. Dynamic convolution (BMM ile)
        mask_logits = self.mask_heads_forward(x, weights, biases)
        return mask_logits  # (N, L, H, W) spatial mask
    
    def mask_heads_forward(self, features, weights, biases):
        x = features.view(N, C, H * W)
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = w.bmm(x) + b  # Batch Matrix Multiply
            if i < n_layers - 1:
                x = F.relu(x)
        return x.view(N, -1, H, W)
```

**Akış:**
```
Decoder Queries (N,L,C) 
    ↓
ctnet_head → params (N,L,num_params)
    ↓
parse_dynamic_params → weights, biases
    ↓
BMM: weights @ features → (N,L,H,W) spatial output
```

---

## Bizim DynamicKernelHead Yapısı

```python
# LSTR_CULANE_2k_mamba_dynamic.py
class DynamicKernelHead(nn.Module):
    def forward(self, queries, feat_map):
        # queries: (N, B, D) decoder output
        # feat_map: (B, C, H, W) layer2
        
        # 1. Kernel generation from queries
        kernels = self.kernel_proj(queries_flat)  # (B*N, D)
        
        # 2. BMM: kernels @ features
        lane_features = torch.bmm(kernels, feat_flat)  # (B, N, H*W)
        
        # 3. Project to 72 points + global pooling
        point_features = self.point_proj(lane_features)
        x_coords = F.adaptive_avg_pool2d(point_features, (1, 1))
        
        return x_coords  # (B, N, 72)
```

**Akış:**
```
Decoder Queries (N,B,D)
    ↓
kernel_proj → kernels (B*N,D)
    ↓
BMM: kernels @ layer2_feat → (B,N,H*W)
    ↓
point_proj + global_pool → (B,N,72) x_coords
```

---

## Temel Farklar

| Özellik | CondLSTR | Bizim DynamicKernelHead |
|---------|----------|-------------------------|
| **Output** | (B, L, H, W) spatial mask | (B, N, 72) points |
| **Feature** | Multi-scale features | Single layer2 |
| **Operation** | Dynamic convolution | BMM + point projection |
| **Param. generation** | weights + biases (dynamic FC) | Kernels only |
| **Post-processing** | Mask-to-points conversion | Direct x_coords |
| **Loss** | Dice/Focal on masks | BCE on x_coords |

---

## CondLSTR Daha Karmaşık

CondLSTR'nin `DynamicMaskHead`:
- **`parse_dynamic_params`**: Parametreleri weight ve bias'e ayırır
- **`mask_heads_forward`**: Multi-layer dynamic convolution
- **Coordinate encoding**: `compute_locations` ile spatial coords ekleniyor
- **Mask interpolation**: Output'da interpolation yapılıyor

Bizimkiler daha basit:
- Tek layer BMM
- Global pooling
- Doğrudan x_coords output

---

## Hangisi Daha İyi?

**CondLSTR Artıları:**
- Spatial output (H×W mask) → daha detaylı
- Multi-layer dynamic convolution → daha güçlü
- Coordinate encoding → spatial awareness

**Bizim DynamicKernelHead Artıları:**
- Daha basit implementasyon
- Daha az parametre
- Doğrudan x_coords → post-processing'e gerek yok
- layer2 kullanımı → early feature'den lane extraction

---

## Sonuç

CondLSTR'nin `DynamicMaskHead` aslında **dynamic convolution** yaklaşımı (CondInst benzeri). Bizim `DynamicKernelHead` ise **kernel-based feature extraction + point projection**.

Farklı yaklaşımlar:
- CondLSTR: Learn to generate dynamic CONV weights → predict spatial masks
- Bizimki: Learn kernels → extract lane features → predict 72 points

Bizim implementasyonumuz CondLSTR paper'daki "Dynamic Kernel" kavramından ilham alıyor ama implementasyon olarak daha basit ve doğrudan x_coords prediction yapıyor.
