# 🧠 MNIST PyTorch Training Project

## 🚀 Project Description
Welcome to the **MNIST PyTorch Training Project**! This repository showcases a complete machine learning workflow for classifying handwritten digits using PyTorch. The code is modular, well-documented, and perfect for learning, experimenting, or building upon. 

## ✨ Features
- **Lab 1**
  - 📦 Loads and preprocesses the MNIST dataset
  - 🧩 Defines a simple feedforward neural network for digit classification
  - 🏋️ Trains and validates the model with detailed logging
  - 🔮 Predicts and logs results for sample images
  - 🛠️ Modular codebase for easy extension
- **Lab 2**
  - 🖼️ Loads and reshapes sample images from the MNIST dataset
  - 📊 Visualizes and saves reshaped images for inspection
  - 🧪 Demonstrates basic data exploration and visualization techniques
  - 📝 Includes extra comments and logging for ML beginners
- **Lab 3**
  - 🤟 Loads and preprocesses American Sign Language (ASL) MNIST-style data
  - 🧠 Builds a convolutional neural network (CNN) for ASL sign classification
  - 🔄 Reshapes and visualizes input data for CNN compatibility
  - 🏋️ Trains and validates the CNN model with detailed logging
  - 🛠️ Modular and beginner-friendly code with extra comments
- **Lab 4**
  - 🎨 Applies data augmentation techniques to ASL images
  - 🔄 Includes random rotations, shifts, flips, and color adjustments
  - 📊 Visualizes original and augmented images for comparison
  - 🚀 Improves model robustness and generalization with augmented training data
- **Lab 5**
  - 🖼️ Loads and displays two sample images (a.png and b.png) from the lab5 dataset
  - 🔍 Demonstrates image loading and visualization using matplotlib
  - 📊 Shows both images side-by-side in a single plot for easy comparison
  - 📝 Useful for understanding image data handling and visualization in ML workflows
- **Lab 6**
  - 🐶 Loads a pretrained VGG16 model for image classification
  - 🖼️ Processes and classifies sample images
  - 📊 Visualizes model predictions and interprets results
  - 📝 Demonstrates transfer learning and model inference with PyTorch

## 🏁 Getting Started
1. ⬇️ Clone the repository.
2. 📦 Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. ▶️ Run the main script:
   ```bash
   python main.py
   ```

## 🎨 Code Style & Rules
- 📝 Follows [PEP8](https://peps.python.org/pep-0008/) guidelines
- 📚 All functions include docstrings and type hints
- 🪵 Logging is used for all major steps and errors
- 🛡️ Exception handling is present for critical operations
- 🧩 Modular structure: data, device, helper, and lab logic are separated

## 🗂️ File Structure
```
main.py                # Entry point for running the labs
labs/                  # All labs
  lab1.py              # Lab 1: MNIST training and evaluation
  lab2.py              # Lab 2: MNIST data visualization and exploration
  lab3.py              # Lab 3: ASL data preprocessing and CNN training
  lab4.py              # Lab 4: Data augmentation for ASL CNN (image transformations, visualization)
  lab5.py              # Lab 5: Image loading and visualization examples
  lab6.py              # Lab 6: Transfer learning with pretrained VGG16 model
tools/                 # Utilities
  data.py              # Data loading utils
  device.py            # Device selection utils
  helper_system.py     # System utils
requirements.txt       # Python dependencies
data/                  # data storage
```

## 📚 Lab Descriptions

### Lab 1: MNIST Training and Evaluation
- Loads MNIST dataset
- Builds and trains a simple CNN
- Evaluates accuracy and visualizes results

### Lab 2: MNIST Data Visualization and Exploration
- Visualizes sample images from MNIST
- Explores data distribution and reshaping
- Saves sample images for review

### Lab 3: ASL Data Preprocessing and CNN Training
- Loads and preprocesses ASL sign language data
- Builds a deeper CNN for multi-class classification
- Trains and validates the model with logging

### Lab 4: Data Augmentation for ASL CNN
- Applies image augmentation techniques (rotation, crop, flip, color jitter)
- Visualizes original and augmented images side-by-side
- Integrates augmentation into training pipeline for improved robustness

### Lab 5: Image Loading and Visualization Examples
- Demonstrates loading and displaying images using matplotlib
- Provides side-by-side comparison of original and augmented images
- Aids in understanding image data handling in machine learning

### Lab 6: Transfer Learning with Pretrained VGG16 Model
- Utilizes a pretrained VGG16 model for image classification
- Processes and classifies sample images (e.g., happy_dog.jpg)
- Visualizes model predictions and interprets results
- Demonstrates transfer learning and model inference with PyTorch

## 🤝 Contributing
- ✨ Follow code style and documentation rules
- 📝 Add docstrings and type hints to new functions
- 🪵 Use logging for all major steps and errors
- 📬 Submit pull requests with clear descriptions

---

> Made with ❤️ and PyTorch. Happy learning!
