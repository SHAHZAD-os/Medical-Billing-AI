# Medical Billing Document Segmentation & Extraction

**Developed by Shahzad Akbar**  
*Deep Learning Solution for Medical Invoice & Billing Document Layout Analysis using Custom PyTorch U-Net.*

---

## 📌 Overview

**Medical Billing Document Segmentation** is a deep learning framework designed to extract, segment, and isolate text regions, tables, invoice headers, and key-value fields from complex medical billing forms, claims (such as CMS-1500 / UB-04), and healthcare invoices.

By leveraging a modular **U-Net** architecture implemented in **PyTorch**, this system transforms unstructured medical documents into pixel-level semantic masks, making downstream Optical Character Recognition (OCR) and automated data extraction fast and highly accurate.

---

## ✨ Key Features

- **Custom Modular U-Net Architecture**: Configurable downsampling/upsampling blocks, base channel widths, and custom scaling factors.
- **Combined Loss Metrics**: Uses a hybrid loss function integrating **Cross-Entropy / Binary Cross-Entropy**, **Dice Coefficient**, and **IoU (Intersection over Union)** for high boundary accuracy.
- **Dataset Preprocessing & Augmentation**: Custom dataset pipeline (`DocumentDataset`) handling image scaling, normalization, and semantic segmentation masks.
- **Interactive Training & Monitoring**: Built-in visual metrics plotting (`plot_summary`) saved automatically to `./runs/` after training runs.
- **Flexible Inference & Evaluation**: Standalone scripts for model evaluation (`evaluate.py`) and fast single-image inference (`inference.py`).

---

## 🏗️ Project Structure

```text
Medical-Billing-AI/
├── data/                  # Dataset utilities & data directory setup
├── models/                # PyTorch neural network architectures (U-Net)
│   └── saves/             # Pretrained & fine-tuned model weights (.pth)
├── scripts/               # Helper scripts (weight downloading, bash scripts)
├── utils/                 # Dataset loaders & image transformation utilities
├── build_dataset.py       # Preprocessing & dataset structuring script
├── train.py               # Main training pipeline with mini-batch SGD & LR scheduler
├── evaluate.py            # Model validation & metrics calculation (Dice / IoU)
├── inference.py           # Test inference script on new medical documents
├── loss.py                # Loss functions (Dice loss, IoU loss)
└── requirements.txt       # Python dependencies
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/SHAHZAD-os/Medical-Billing-AI.git
cd Medical-Billing-AI
pip install -r requirements.txt
```

### 2. Dataset Preparation

Organize your training data with corresponding mask images:

```bash
python build_dataset.py --input_dir ./raw_documents --output_dir ./data
```

### 3. Model Training

Train the U-Net model with combined Dice + IoU loss enabled:

```bash
python train.py \
  --train_data_paths ./data/train/images ./data/train/masks \
  --validation_data_paths ./data/val/images ./data/val/masks \
  --save_name medical_unet_v1 \
  --num_epochs 25 \
  --batch_size 8 \
  --learning_rate 3e-4 \
  --use_dice_and_iou \
  --verbose
```

### 4. Running Inference

To generate segmentation masks for target medical invoices or billing documents:

```bash
python inference.py \
  --model_path ./models/saves/medical_unet_v1.pth \
  --input_image ./samples/sample_bill.png \
  --output_path ./samples/output_mask.png
```

---

## 📊 Evaluation & Metrics

The model optimizes both region alignment and boundary precision using:

$$ \text{Total Loss} = \text{BCE Loss} + \text{Dice Loss} + \text{IoU Loss} $$

- **Dice Coefficient**: Measures overlap accuracy for segmented billing regions.
- **IoU (Jaccard Index)**: Ensures tight bounding masks around small text boxes and numerical billing fields.

---

## 👨‍💻 Author

**Shahzad Akbar**  
- GitHub: [@SHAHZAD-os](https://github.com/SHAHZAD-os)  
- Expertise: AI/ML, Computer Vision, PyTorch, Document Parsing  
