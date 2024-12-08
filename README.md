
# Activation Functions and their Importance Tutorial

This repository contains the code used to demonstrate the effects of using activation functions in a Neural Network model and how different activation Functions works in neural networks. Follow the steps below to set up the project and reproduce the results.

---

## **Getting Started**

### **1. Clone or Download the Repository**

To begin, download the repository from GitHub. You can either:  
- **Clone the repository** using Git:
  ```bash
  git clone https://github.com/Rudr-krishna/Neural_Netwok_Tutorial.git
  cd your-repo-name
  ```
- **Download the ZIP file**:
  - Go to the repository page on GitHub.
  - Click the green **Code** button.
  - Select **Download ZIP**.
  - Extract the ZIP file to your desired location.

---

### **2. Set Up the Environment**

Ensure you have Python installed on your system (preferably version 3.7 or higher).

#### **Create a Virtual Environment (Optional but Recommended)**

1. Create a virtual environment:
   ```bash
   python -m venv env
   ```
2. Activate the virtual environment:
   - **Windows:**
     ```bash
     .\env\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source env/bin/activate
     ```

#### **Install Required Libraries**

Install the dependencies listed in `requirements.txt` using pip:
```bash
pip install -r requirements.txt
```

---

### **3. Run the Code**

After setting up the environment, run the provided Python script to train the models and generate results.

1. Navigate to the project directory if not already there:
   ```bash
   cd your-repo-name
   ```
2. Run the scripts:
   ```bash
   python Code/With_and_without_Activation.py (For the comparison of having a neural network with and without Activation Function.)
   ```
   ```bash
   python Code/Comparison_between_Acti_Functions.py (For the comparison of different Activation Functions.)
   ```   

---

### **4. Reproduce Results**

- The script will train models with different activation Functions (`Sigmoid`, `Tanh`, and `ReLU`) and display validation loss and accuracy results for comparison.
- Graphs such as *Validation Loss Comparison* and *Training vs. Validation Accuracy* will be the outputs.

---

## **Requirements**

The `requirements.txt` file includes the following essential libraries:
- TensorFlow/Keras
- Matplotlib
- NumPy
- Torch

---

## **Contributing**

Contributions are welcome! Feel free to fork the repository, submit issues, or make pull requests.

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
