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
tools/                 # Utilities
  data.py              # Data loading utils
  device.py            # Device selection utils
  helper_system.py     # System utils
requirements.txt       # Python dependencies
data/                  # data storage
```

## 🤝 Contributing
- ✨ Follow code style and documentation rules
- 📝 Add docstrings and type hints to new functions
- 🪵 Use logging for all major steps and errors
- 📬 Submit pull requests with clear descriptions

---

> Made with ❤️ and PyTorch. Happy learning!
