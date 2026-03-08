# clrernet ortamını yeniden kurma (conda create + pip install)

Bu dosya, mevcut `clrernet` ortamındaki sürümleri temel alır ve sunucuda adım adım kurulacak şekilde hazırlanmıştır.

## 1) Conda ortamını oluştur ve aktive et

```bash
conda create -n clrernet python=3.10.18 pip=25.2 -y
conda activate clrernet
python -m pip install --upgrade pip
```

## 2) PyTorch (CUDA 11.8) — önce bunu kur

```bash
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## 3) Diğer paketler (tek tek, sürümlü)

```bash
pip install absl-py==2.3.1
pip install addict==2.4.0
pip install albumentations==0.4.6
pip install aliyun-python-sdk-core==2.16.0
pip install aliyun-python-sdk-kms==2.16.5
pip install certifi==2025.8.3
pip install cffi==2.0.0
pip install charset-normalizer==3.4.3
pip install click==8.3.0
pip install colorama==0.4.6
pip install contourpy==1.3.2
pip install crcmod==1.7
pip install cryptography==46.0.1
pip install cycler==0.12.1
pip install dill==0.4.0
pip install exceptiongroup==1.3.1
pip install filelock==3.14.0
pip install fonttools==4.60.0
pip install fsspec==2024.6.1
pip install grpcio==1.75.1
pip install h5py==3.15.1
pip install idna==3.10
pip install imageio==2.37.0
pip install imgaug==0.4.0
pip install iniconfig==2.3.0
pip install Jinja2==3.1.4
pip install jmespath==0.10.0
pip install joblib==1.5.2
pip install kiwisolver==1.4.9
pip install lazy_loader==0.4
pip install Markdown==3.9
pip install markdown-it-py==4.0.0
pip install MarkupSafe==2.1.5
pip install matplotlib==3.10.6
pip install mdurl==0.1.2
pip install mmcv==2.1.0
pip install mmdet==3.3.0
pip install mmengine==0.8.4
pip install model-index==0.1.11
pip install mpmath==1.3.0
pip install multiprocess==0.70.18
pip install networkx==3.3
pip install numpy==1.24.3
pip install opencv-python==4.12.0.88
pip install opencv-python-headless==4.11.0.86
pip install opendatalab==0.0.10
pip install openmim==0.3.9
pip install openxlab==0.1.2
pip install ordered-set==4.1.0
pip install oss2==2.17.0
pip install p-tqdm==1.4.2
pip install packaging==24.2
pip install pandas==2.3.2
pip install pathos==0.3.4
pip install pillow==11.3.0
pip install platformdirs==4.4.0
pip install pluggy==1.6.0
pip install pox==0.3.6
pip install ppft==1.7.7
pip install protobuf==6.32.1
pip install pycocotools==2.0.10
pip install pycparser==2.23
pip install pycryptodome==3.23.0
pip install Pygments==2.19.2
pip install pyparsing==3.2.5
pip install pytest==9.0.2
pip install python-dateutil==2.9.0.post0
pip install pytz==2023.4
pip install PyYAML==6.0.3
pip install qudida==0.0.4
pip install regex==2025.9.18
pip install requests==2.28.2
pip install rich==13.4.2
pip install scikit-image==0.25.2
pip install scikit-learn==1.7.2
pip install scipy==1.15.3
pip install shapely==2.1.2
pip install six==1.17.0
pip install sympy==1.13.3
pip install tabulate==0.9.0
pip install tensorboard==2.20.0
pip install tensorboard-data-server==0.7.2
pip install termcolor==3.1.0
pip install terminaltables==3.1.10
pip install threadpoolctl==3.6.0
pip install tifffile==2025.5.10
pip install tomli==2.2.1
pip install tqdm==4.65.2
pip install typing_extensions==4.15.0
pip install tzdata==2025.2
pip install ujson==5.11.0
pip install urllib3==1.26.20
pip install Werkzeug==3.1.3
pip install yapf==0.43.0
```

## 4) Ortama bağlı / opsiyonel satırlar

`pip_list.txt` içinde aşağıdaki iki özel durum da vardı:

```bash
# Sadece Windows için (Linux sunucuda kurmayın):
pip install pywin32==311

# Lokal editable paket (sunucuda path'i kendi yoluna uyarlayın):
pip install -e /path/to/clrernetlm
```

