🌱 Plant Disease Image Classifier

This project is a Convolutional Neural Network (CNN) based image classifier that detects bean leaf health status:

✅ Healthy

🍂 Bean Rust

⚡ Angular Leaf Spot

The model is trained on the Beans Dataset (from Hugging Face / TensorFlow Datasets) and can predict plant health from a single image.

📂 Project Structure
plant-disease-classifier/
│── dataset/
│   ├── train/
│   │   ├── healthy/
│   │   ├── bean_rust/
│   │   └── angular_leaf_spot/
│   ├── test/
│   │   ├── healthy/
│   │   ├── bean_rust/
│   │   └── angular_leaf_spot/
│   └── validation/   (optional)
│
│── plant_classifier.py   # Training script
│── plant_model.h5        # Saved trained model
│── predict_image.py      # Script for single image prediction
│── requirements.txt
│── README.md

⚙️ Requirements

Install dependencies:

pip install -r requirements.txt


requirements.txt

tensorflow
matplotlib
scikit-learn
numpy

🚀 Training the Model

Run the training script:

python plant_classifier.py


This will:

Load dataset (train/test split)

Train CNN for 10–15 epochs

Evaluate model on test data

Plot accuracy & loss curves

Save trained model as plant_model.h5

🖼️ Predicting a Single Image

Use the predict_image.py script to test with one image:

python predict_image.py


Inside predict_image.py, update the path to your test image:

predict_image("dataset/test/healthy/abc.jpg")


Sample Output:

✅ Prediction: healthy (Confidence: 0.97)

📊 Model Evaluation

Accuracy on test set

Confusion matrix

Classification report

Training & validation curves

📌 Dataset

Beans Dataset (Healthy, Bean Rust, Angular Leaf Spot)

Available on Hugging Face
 or TensorFlow Datasets.

📦 Deliverables

plant_classifier.py → Training code

plant_model.h5 → Saved model

predict_image.py → Script for prediction

requirements.txt → Dependencies

README.md → Documentation
