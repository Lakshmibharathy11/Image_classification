# 🖼️ Malicious vs Benign Image Classification  

🚀 *San Jose State University – Final Project (Group 4)*  
👩‍💻 Team: Lakshmi Bharathy Kumar, Shao-Yu Huang, Yi-An Yao, Yilin Sun  

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)]() 
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)]() 
[![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-231F20?logo=apache-kafka&logoColor=white)]() 
[![Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?logo=apachespark&logoColor=white)]()  

---

## 📌 Abstract  
This project develops a **real-time harmful content detection system** using deep learning.  
- Fine-tuned **MobileNetV2** and **ResNet50** on a curated subset of **Open Images Dataset V7**.  
- Built a **real-time streaming pipeline** with **Apache Kafka** and **Spark** to simulate image uploads.  
- Achieved **83.5% accuracy with MobileNetV2** and **82.2% with ResNet50**, demonstrating the potential for scalable **real-time content moderation** in social media and safety-critical applications.  

---

## 🧐 Introduction  
With the surge in **user-generated visual content**, harmful images (weapons, violence, explosions) pose risks to safety and platform integrity. Manual moderation is not scalable.  
👉 Our system builds a **real-time image classification pipeline** to distinguish between **harmful** and **benign** images, combining **lightweight CNNs** with **streaming architectures** for efficiency.  

---

## 📊 Dataset  
- **Source:** Google’s *Open Images Dataset V7* (9M images, 19,995 categories)  
- **Subset:** Harmful (e.g., gun, knife, explosion) vs Safe (e.g., person, dog, tree)  
- **Challenge:** Severe class imbalance → solved by **manual selection** for binary classification  
- **Preprocessing:**  
  - Resize to 224×224  
  - Normalization with ImageNet mean/std  
  - Balanced train/validation splits  

---

## ⚙️ Methodology  

### 🧠 Models  
- **MobileNetV2** → lightweight, efficient, suitable for real-time & edge devices  
- **ResNet50** → deeper architecture, higher representational power  

### 🔄 Streaming Pipeline  
- Images ingested via **Kafka Producer** → sent to `image_data` topic  
- **Kafka Consumer** uses PyTorch model for classification  
- Output includes **image metadata + predicted class**  
- Achieved **<300ms latency** on CPU-only environment  

---

## 📈 Results  

### ✅ Model Performance  
| Model       | Accuracy | Precision (Harmful) | Recall (Harmful) | F1-score (Harmful) |
|-------------|----------|---------------------|------------------|--------------------|
| MobileNetV2 | **83.5%** | 0.87                | **0.79**         | 0.83               |
| ResNet50    | 82.2%    | **0.96**            | 0.68             | 0.79               |

- **MobileNetV2** → Better **recall** (captures more harmful cases, fewer misses)  
- **ResNet50** → Better **precision** (fewer false alarms, but more misses)  

### 📊 Additional Metrics  
- ROC-AUC: 0.92 (MobileNetV2), 0.93 (ResNet50)  
- Confusion Matrix: MobileNetV2 misclassified 146 harmful images vs ResNet50’s 221  

---

## 📢 Discussion  
- **MobileNetV2** is better suited for safety-critical applications → *missing harmful content is worse than false positives*.  
- Kafka integration shows strong potential for **real-time AI pipelines**.  
- Applications:  
  - Social media content filtering  
  - Live surveillance monitoring  
  - Automated moderation tools  
  - Real-time categorization (e.g., fashion, e-commerce)  

---

## 🔮 Future Work  
- Support **remote image URLs** & **cloud object storage (S3/GCS)**  
- Deploy on **GPU/Edge devices** for lower latency  
- Extend to **multi-class harmful content classification**  
- Integrate with streaming engines like **Apache Flink / Spark Streaming**  

---

## 📬 References  
- [Open Images Dataset V7](https://storage.googleapis.com/openimages/web/download_v7.html)  
- [Demo Images](https://drive.google.com/drive/folders/1404MArZCvH78d9UxaUGctLcUZ1rQjhah?usp=sharing)  
- [Trained Models](https://drive.google.com/drive/folders/1d6dlMSfdFAvN5FRN9hJIU55pWAO-H4-U?usp=drive_link)  
- [Training/Testing Dataset](https://drive.google.com/drive/folders/15_WRlkWXn0nEOQSQ3Ct_1LLsbo6UpELU?usp=driv_link)  

---
