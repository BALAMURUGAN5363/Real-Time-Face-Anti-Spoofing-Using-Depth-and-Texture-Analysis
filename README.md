# 🎯 Real-Time Face Anti-Spoofing Using Depth & Texture Analysis

An **AI-powered real-time face liveness detection system** designed to prevent spoofing attacks such as **photo attacks, video replay attacks, and mask attacks**.  

This project implements a **dual deep learning architecture** combining **Texture (RGB) features** and **Depth (3D structural) features** for robust and secure authentication.

---

## 🧠 Key Features

- 🎥 Real-time webcam-based face detection  
- 📸 Detects photo spoof attacks  
- 📱 Detects video replay attacks  
- 🎭 Detects mask-based attacks  
- 🧠 Dual-model architecture (Texture + Depth)  
- 🔗 Feature-level fusion for improved accuracy  
- 📊 Accuracy, Loss & ROC curve evaluation  
- ⚖️ Class imbalance handling  
- ⚡ Fast and lightweight inference  

---

## 🛠 Tech Stack

### 🖥 Frontend (Optional Deployment)
- HTML  
- CSS  
- JavaScript  
- Angular (if integrated)  
- Bootstrap / Custom UI  

### ⚙️ Backend
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## 🧠 Machine Learning Architecture

- Convolutional Neural Networks (CNN)  
- Texture Model (RGB-based feature extraction)  
- Depth Model (Depth-map based feature extraction)  
- Feature Fusion Model  
- Binary Classification (Real vs Spoof)  

---
---

## 🏗 System Architecture

```
+------------------+
|   Webcam Input   |
+------------------+
          │
          ▼
+--------------------------------+
| Face Detection & Preprocessing |
+--------------------------------+
          │
          ▼
+-----------------------+     +-----------------------+
|   Texture CNN Model   |     |    Depth CNN Model    |
|    (RGB Features)     |     |   (Depth Features)    |
+-----------------------+     +-----------------------+
          │                             │
          └───────────────┬─────────────┘
                          ▼
                +--------------------+
                |   Feature Fusion   |
                |  (Concatenation)   |
                +--------------------+
                          │
                          ▼
                +--------------------+
                |  Fully Connected   |
                |     Classifier     |
                +--------------------+
                          │
                          ▼
                +--------------------+
                |     Prediction     |
                |    ✅ REAL FACE    |
                |    ❌ SPOOF ATTACK |
                +--------------------+
```

### 🔄 Workflow Summary

1. Webcam captures live video  
2. Face region is detected and preprocessed  
3. Texture CNN extracts surface-level features  
4. Depth CNN extracts 3D structural features  
5. Features are concatenated (fusion)  
6. Final classifier predicts **Real vs Spoof**

---

## 📊 Model Performance

| Model           | Accuracy |
|---------------|----------|
| Texture Model | ~87%     |
| Depth Model   | ~91%     |
| 🔥 Fusion Model | **~93% (Best Performance)** |

### 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC Curve  
- Confusion Matrix  

---

## 🔐 Real-World Applications

- 🔐 Banking & Financial Authentication  
- 🏢 Secure Office Entry Systems  
- 📱 Mobile Face Unlock Systems  
- 🛂 Border & Surveillance Systems  
- 🧑‍💻 Online Exam Proctoring  

---

## 🚀 Future Enhancements

- 🔬 Transformer-based backbone (Vision Transformer - ViT)  
- 🧠 Attention-based feature fusion  
- 📱 Mobile deployment optimization  
- ☁️ REST API deployment  
- 🎯 Grad-CAM explainability  
- 🌍 Multi-dataset generalization testing  

---

## 👨‍💻 Author

### Bala  
🎓 **B.Tech – Artificial Intelligence & Data Science**  
🔍 Passionate about **AI Security, Deep Learning & Computer Vision**

---

### 📫 Connect With Me
- 💼 LinkedIn: *(https://www.linkedin.com/in/balamurugan-s-b28635255)*
- 📧 Email: *(balamurugan.s6f@gmail.com)*

---

> 🚀 Building intelligent and secure AI systems for real-world applications.

