# LSTR + Mamba Sunucu Kurulum Rehberi

Bu rehber, lokaldeki Windows/WSL `clrernet` ortamınızın birebir aynısını (aynı Python, PyTorch ve spesifik paket sürümleriyle) yeni Linux sunucuda kurmanızı sağlar. Mamba entegrasyonu için gereken CUDA ve derleme ayarları da dahildir.

---

## 1. Conda Kurulumu (Eğer Sunucuda Yoksa)
Eğer sunucuda Miniconda/Anaconda yüklü değilse, aşağıdaki blok ile kurabilirsiniz:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

---

## 2. Ortamı Oluşturma ve Temel Paketleri Yükleme

Lokaldeki sisteminizin tam spesifikasyonuna göre:
- **Python**: 3.10.18
- **PyTorch**: 2.0.1+cu118
- **Torchvision**: 0.15.2+cu118

```bash
# 1. 'clrernet' adında conda ortamı oluştur
conda create -n clrernet python=3.10.18 -y
conda activate clrernet

# 2. PyTorch ve CUDA 11.8 Toolkit (Lokaldeki aynı uyum)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# 3. İsteğe Bağlı: NVCC derleyici için Conda cudatoolkit (Mamba derlemesi için gerekebilir)
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
```

---

## 3. LSTR (OpenMMLab ve Temel Bağımlılıklar)

OpenMMLab (MMCV, MMDet) paketleri, LSTR için önemlidir. Sürüm çakışması yaşamamak için MIM aracını veya doğrudan Wheel'leri kullanmalısınız. Sizin ortamınızda kurulu olan birebir sürümler:

```bash
# 1. MIM Kurulumu
pip install -U openmim==0.3.9

# 2. MMCV Stack (Sizin conda ortamında tam olarak bu versiyonlardı)
mim install mmengine==0.8.4
mim install "mmcv==2.1.0"
mim install "mmdet==3.3.0"

# 3. Klasik Bağımlılıklar
pip install numpy==1.24.3
pip install scipy==1.15.3
pip install h5py==3.15.1
pip install ujson==5.11.0
pip install opencv-python==4.12.0.88
pip install opencv-python-headless==4.11.0.86
pip install tqdm==4.65.2
pip install tensorboard==2.20.0
pip install thop  # MACs / Param hesabı için (sürüm belirtilmez genelde son sürüm iyidir)
```

*(Gerektiğinde diğer küçük paketleri `pip install addict albumentations imageio matplotlib pandas pillow pyyaml scikit-image scikit-learn shapely` şeklinde yükleyebilirsiniz.)*

---

## 4. Mamba & Causal Conv1d Kurulumu (Kritik Adım)

`mamba-ssm` C++ CUDA kernel'ları derlerken çok katı davranır. PyTorch `2.0.1+cu118` ile çalışması en stabil olan yöntem teker teker kurmaktır.

```bash
# Önce Causal Conv1d
pip install causal-conv1d==1.1.1 --no-build-isolation

# Sonra Mamba (Derlenmesi 5-10 dakika sürebilir)
pip install mamba-ssm==1.1.1 --no-build-isolation
```

> **Not:** Kurulum sırasında hata alırsanız, sunucuda uygun `gcc/g++` olduğundan emin olun (`sudo apt install build-essential`). Ayrıca Linux'taki `nvcc` ile PyTorch'un CUDA'sının (11.8) aynı hizaya geldiğinden emin olmak için `nvcc --version` kontrol edin.

---

## 5. Eğitimi Başlatma

Verisetinizin CULane klasör yapısında (`/home/kullanici/projects/CULane` gibi) ve klasör izinlerinin düzgün (`chmod -R 755`) olduğundan emin olun.

Proje dizininde (LSTR):

```bash
# Veri dizini LSTR_CULANE_MAMBA.json içinde '/home/alki/projects/' olarak ayarlı.
# Bunu sunucudaki kendi dizininize göre GÜNCELLEMEYİ UNUTMAYIN! (örneğin /data/projects/)

conda activate clrernet

# Ana Mamba Eğitimi
python train.py LSTR_CULANE_MAMBA

# Başladıysa arka planda bırakmak için nohup veya tmux kullanın:
# nohup python train.py LSTR_CULANE_MAMBA > training.log 2>&1 &
```
