🎯 Real-Time Face Anti-Spoofing Using Depth & Texture Analysis

An AI-powered real-time face liveness detection system that prevents spoofing attacks such as photo, video replay, and mask attacks. This project uses a dual deep learning architecture combining texture and depth features for robust security authentication.

🧠 Features

🎥 Real-time webcam-based face detection

📸 Detects photo spoof attacks

📱 Detects video replay attacks

🎭 Detects mask-based attacks

🧠 Dual-model architecture (Texture + Depth)

🔗 Feature-level fusion for improved accuracy

📊 Accuracy, Loss, ROC curve evaluation

⚖️ Class imbalance handling

⚡ Fast and lightweight inference

🛠 Tech Stack
🖥 Frontend (Optional Deployment)

HTML, CSS, JavaScript

Angular (if integrated)

Bootstrap / Custom UI

⚙️ Backend

Python

TensorFlow / Keras

OpenCV

NumPy

Scikit-learn

Matplotlib

🧠 Machine Learning

Convolutional Neural Networks (CNN)

Texture Model (RGB-based)

Depth Model (Depth-map based)

Feature Fusion Model

Binary Classification (Real vs Spoof)

🏗 System Architecture

Webcam captures live video

Face region extracted & preprocessed

Texture model extracts surface-level features

Depth model extracts structural 3D features

Feature vectors concatenated

Fusion classifier predicts:

✅ REAL FACE

❌ SPOOF ATTACK

📊 Model Performance

Texture Model Accuracy: ~87%

Depth Model Accuracy: ~91%

Fusion Model Accuracy: ~93% (Best Performance)

Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

ROC Curve

Confusion Matrix
🔐 Real-World Applications

🔐 Banking & Financial Authentication

🏢 Secure Office Entry Systems

📱 Mobile Face Unlock Systems

🛂 Border & Surveillance Systems

🧑‍💻 Online Exam Proctoring

🚀 Future Enhancements

🔬 Transformer-based backbone (ViT)

🧠 Attention-based feature fusion

📱 Mobile deployment

☁️ REST API deployment

🎯 Grad-CAM explainability

🌍 Multi-dataset generalization testing

👨‍💻 Author

Bala
B.Tech Artificial Intelligence & Data Science
Focused on AI Security, Deep Learning & Computer Vision
