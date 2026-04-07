# 🚢 Maritime Anomaly Intelligence: Unsupervised Vessel Detection

**Real-time maritime surveillance using Convolutional Autoencoders (CAE) on high-resolution satellite imagery.**

---

## 📌 Project Overview

Global maritime security is often compromised by **"dark vessels"** — ships that disable tracking systems to engage in illegal activities.

This project introduces an **unsupervised anomaly detection system** that learns the natural patterns of ocean surfaces and detects any deviation as a potential vessel.

Instead of relying on labeled ship data, the model identifies **statistical anomalies**, making it effective for **zero-day detection scenarios**.

---

## ✨ Key Features

- 🧠 **Unsupervised Learning (CAE)**
  - No labeled dataset required
  - Learns "normal ocean patterns"

- 📉 **88.05% Error Reduction**
  - Compared to traditional dense models

- 📊 **Statistical Thresholding**
  - 95th percentile rule for anomaly detection  
  - Threshold: `0.001246`

- 🎥 **Multi-Input Support**
  - Webcam (real-time)
  - Image upload
  - Video input

- 📊 **Real-Time Dashboard**
  - MSE heatmaps
  - Bounding box detection
  - Inference time: ~0.75 seconds

---

## 🏗️ System Architecture

The model follows an **Encoder–Decoder architecture**:

1. **Input**
   - 80 × 80 RGB satellite image patches

2. **Encoder**
   - Extracts features using Conv2D + ReLU + MaxPooling
   - Compresses into latent space (128 units)

3. **Decoder**
   - Reconstructs the image from compressed representation

4. **Anomaly Detection**
   - Calculates **Mean Squared Error (MSE)**
   - High error → anomaly (possible vessel)

---

## 🛠️ Tech Stack

- **Language**: Python 3.10  
- **Deep Learning**: TensorFlow, Keras  
- **Frontend**: Streamlit  
- **Libraries**: OpenCV, NumPy, Scikit-learn  
- **Platform**: Kaggle / Google Colab (GPU)

---

## ⚙️ Installation & Usage

### 🔹 Prerequisites
- Python 3.10+
- Git installed

### 🔹 Setup

```bash
# Clone repository
git clone https://github.com/YourUsername/Maritime-Anomaly-Detection.git

cd Maritime-Anomaly-Detection

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py

| Metric         | Baseline (Dense) | Proposed (CAE) | Improvement      |
| -------------- | ---------------- | -------------- | ---------------- |
| MSE Loss       | 0.011356         | **0.001246**   | **88.05%**       |
| Inference Time | 1.2s             | **0.75s**      | **37.5% Faster** |
