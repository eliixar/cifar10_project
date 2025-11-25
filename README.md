# COMP472 Project – CIFAR-10 Image Classification
Written by: Alexa Rokakis
40234010

github repository: https://github.com/eliixar/cifar10_project
This repository contains the complete implementation of the COMP472 Fall 2025 course project.  
The project performs **image classification on CIFAR-10** using four main AI models:

# **DEPENDENCIES**
These are important as the code will not run without installing them.
They can be found in requirements.txt
To install all necessary modules, run the following in the terminal:

pip install -r requirements.txt

OR

pip install numpy matplotlib joblib scikit-learn torch torchvision

## Models
1. Gaussian Naive Bayes (custom + Scikit-Learn)
2. Decision Tree (custom + Scikit-Learn)
3. Multi-Layer Perceptron (base + depth experiment + width experiment)
4. Convolutional Neural Network (VGG11 + depth variants + kernel size variants)

All evaluation metrics requested are included:
- Accuracy
- Precision
- Recall
- F1-measure  
- Confusion Matrix for **every model and every variant**

---

# **Repository Contents**

project/
│── run_all.py # Main execution script (trains + loads + evaluates everything)
│── README.md # This file
│── features/ # ResNet + PCA features (50×1 vectors)
│ ├── X_train_50.npy
│ ├── y_train.npy
│ ├── X_test_50.npy
│ └── y_test.npy
│
│── saved_models/ # Saved model files (all models + variants)
│
│── confusion_matrices/ # PNG confusion matrices for all experiments
│
│── cifar_loader.py # load raw images for VGG11
│
└── src/
└── models/ # folder containing all models used
├── naive_bayes.py # naive bayes that i coded
├── decision_tree.py # decision tree that i coded
├── sklearn_decision_tree.py # sci-kit learn's decision tree
├── mlp_main.py # mlp that i coded
├── mlp_depth_experiment.py # mlp depth experiment file
├── mlp_width_experiment.py # mlp width experiment (hidden layers)
├── vgg11.py # vgg-11 fed on raw images
├── vgg_depth_experiment.py # vgg11 depth experiment
└── vgg_kernels_experiment.py # vgg11 kernel experiment


---

#  **1. Data Pre-Processing Instructions**

### Required before running the models
Extract CIFAR-10 features using:
- Pretrained ResNet-18 (image resized to 224×224 and normalized)
- Removing last FC layer
- Output feature vector: **512 × 1**
- Apply PCA → reduce to **50 × 1**

These have already been pre-computed and stored in:
features/X_train_50.npy
features/y_train.npy
features/X_test_50.npy
features/y_test.npy

If these files are missing, the code will NOT run.

# **2. Running All Models Easily**

### ➤ To train, load, and evaluate **every single model and experiment**, run:
python run_all.py

This is the only file needed to run. It automatically trains, loads, saves and evaluates
every model and their variations within the main function.

All saved models go to "saved_models/"
All confusion matrices go to "confusion_matrices/"

# **3. Outputs Produced**

After running `run_all.py`, the following are generated:

### Saved Model Files  
Every major model + variant has a ".pth", ".pt" or ".pkl".

### Confusion Matrices  
Each model and experiment produces:

- A printed confusion matrix  
- A saved PNG in "confusion_matrices" folder, each with an appropriate file name

For each model, these metrics are evaluated and printed, with a summary at the end of each section:
- Accuracy  
- Precision  
- Recall  
- F1 Score  

---

 **4. How to Train and Evaluate Each Model Manually**

### **Naive Bayes**

from src.models.naive_bayes import GaussianNaiveBayes


### **Decision Tree**

from src.models.decision_tree import DecisionTreeClassifier


### **MLP**

from src.models.mlp_main import train_mlp, predict_mlp


### **VGG11**

from src.models.vgg11 import load_or_train_vgg11, evaluate_vgg11


---

# Performance Notes (for marker)
- VGG11 is trained according to assignment rules  
  (SGD optimizer + CrossEntropyLoss + correct architecture)
- Training can be slow on CPU. Model will load from disk if already trained
- Epoch count is configurable in `vgg11.py`

---

# **5. Instructions for TAs / Markers**

### To fully reproduce our results:
1. Ensure `features/` contains the required PCA files.  
2. Ensure `saved_models/` is writable.  
3. Run: python run_all.py
4. Compare:
   - Confusion matrices inside `confusion_matrices/`
   - Printed metrics in terminal
   - Saved models in `saved_models/`

