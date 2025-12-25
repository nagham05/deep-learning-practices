# TensorFlow Deep Learning Collection

A comprehensive suite of Jupyter notebooks documenting the journey from basic tensor operations to building high-performance Convolutional Neural Networks (CNNs). This repository serves as both a learning resource and a performance benchmark for standard computer vision datasets.

## 🚀 Getting Started

### Prerequisites

The notebooks are built using **TensorFlow 2.x** and **Keras**. Ensure you have the following installed:

* Python 3.8+
* TensorFlow
* NumPy & Pandas
* Matplotlib (for visualizations)

### Installation

```bash
pip install -r requirements.txt

```

---

## 📚 Notebooks Detailed Overview

### 1. Fundamentals: Introduction to TensorFlow

**File:** `intro_to_tensorflow.ipynb`

This notebook establishes the mathematical foundations of deep learning using the TensorFlow framework.

* **Key Concepts:** * **Tensor Manipulation:** Usage of `tf.constant` and `tf.Variable` for building computational graphs.
* **Mathematical Operations:** Demonstrates linear operations ().


* **Manual Neural Network implementation:** * Hard-coded forward pass logic using `numpy` to replicate how a single neuron functions.
* **Activation Functions Covered:** Sigmoid, ReLU, and Tanh.


* **Results:**
  * **Calculation Output:** A sample operation with  correctly yields a result of `25.0`.



### 2. Basic Classification: Fashion MNIST

**File:** `fashion_mnist_classification.ipynb`

A hands-on guide to classifying 70,000 grayscale images of clothing items into 10 distinct categories (T-shirts, trousers, sneakers, etc.).

* **Model Architecture:** * **Input Layer:** `Flatten` (28x28 pixels to 784).
* **Hidden Layer:** `Dense` (128 units, ReLU activation).
* **Output Layer:** `Dense` (10 units, Softmax activation).


* **Preprocessing:** Pixel values are normalized from the range `[0, 255]` to `[0, 1]` to improve convergence speed.
* **Key Results:**
  * **Prediction:** The model successfully predicts an "Ankle Boot" (Label 9) from the test set.
  * **Environment:** Runs on TensorFlow version `2.20.0`.



### 3. Advanced Vision: CIFAR-10 with CNNs

**File:** `cnn_tf.ipynb`

This notebook transitions from simple dense networks to **Convolutional Neural Networks (CNNs)**, significantly improving accuracy on color image datasets.

* **Dataset:** **CIFAR-10**, consisting of 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, etc.).
* **Model Architecture:**
* **Feature Extraction:** Three `Conv2D` layers paired with `MaxPooling2D` to capture spatial hierarchies.
* **Classification:** A `Dense` head with 64 units leading to the 10-class output.


* **Training Performance:**
  * The model achieves a high validation accuracy, reaching approximately **99.35%** on the training set in the final recorded cells.
  * **Visualization:** Includes plotting for training vs. validation accuracy to detect overfitting.
  


---

## 📊 Summary of Results

| Notebook | Dataset | Model Type | Key Metric |
| --- | --- | --- | --- |
| `intro_to_tensorflow` | Synthetic | Manual Math | Result: 25.0 |
| `fashion_mnist` | Fashion MNIST | Simple Dense | Prediction Match: Label 9 |
| `cnn_tf` | CIFAR-10 | CNN | ~99% Train Accuracy |

---

## 🛠 Usage

Each notebook is self-contained. You can run them in order to see the progression from manual calculations to automated deep learning pipelines.

```python
# Example of loading data in the notebooks
from tensorflow.keras import datasets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

```
