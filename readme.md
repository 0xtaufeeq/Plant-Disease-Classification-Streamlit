# 🌿 Plant Disease Classification using PyTorch and Streamlit

This project implements a deep learning model using PyTorch and ResNet-34 with transfer learning to classify plant leaf images into three categories: Healthy, Powdery Mildew, and Rust. It also includes a Streamlit web application for easy interaction, allowing users to upload leaf images and get predictions.

## Features

*   **Multi-class Classification:** Classifies plant leaves into Healthy, Powdery Mildew, and Rust categories.
*   **Deep Learning Model:** Utilizes a ResNet-34 Convolutional Neural Network (CNN).
*   **Transfer Learning:** Leverages weights pre-trained on the ImageNet dataset for faster convergence and better performance.
*   **Interactive Web UI:** Built with Streamlit for easy image uploads and viewing classification results.
*   **Pre-trained Model:** Loads a pre-trained model (`plant_disease_model.pth`) by default for immediate use.
*   **On-Demand Training:** Provides an option within the Streamlit app to retrain the model if needed (e.g., first run, or to experiment with more epochs).
*   **Data Augmentation:** Uses Albumentations library for augmenting training data (flips) to improve model robustness.

## Technology Stack

*   **Python:** Core programming language.
*   **PyTorch:** Deep learning framework for model building and training.
*   **Torchvision:** Provides access to datasets, models (like ResNet), and image transformations.
*   **Streamlit:** Framework for building the interactive web application.
*   **Albumentations:** Library for efficient image augmentation.
*   **NumPy:** Fundamental package for scientific computing.
*   **Pillow (PIL):** Python Imaging Library for image manipulation.
*   **Kaggle API:** Used for downloading the dataset (optional, if not present locally).
*   **Seaborn / Matplotlib:** For plotting training history (optional display in Streamlit after training).

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/0xtaufeeq/Plant-Disease-Classification-Streamlit.git
    cd Plant-Disease-Classification-Streamlit
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Kaggle API Credentials (for Dataset Download):**
    *   If you don't have the dataset locally in a `data` subdirectory, the application will attempt to download it from Kaggle.
    *   This requires your Kaggle API credentials. Download your `kaggle.json` file from your Kaggle account settings (`Account` -> `API` -> `Create New API Token`).
    *   Place the downloaded `kaggle.json` file in the root directory of this project.
    *   **Security Note:** Ensure your `kaggle.json` file is not committed to public repositories (add it to your `.gitignore` file).

## Dataset

*   **Source:** The project uses the [Plant Disease Recognition Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset) from Kaggle by Rashik Rahman Pritom.
*   **Classes:** Healthy, Powdery, Rust.
*   **Structure:** The dataset should be organized into `Train`, `Validation`, and `Test` subdirectories within a main `data` folder at the project root. Like this:
    ```
    Plant-Disease-Classification-Streamlit/
    ├── data/
    │   ├── Test/
    │   │   ├── Test/
    │   │   │   ├── Healthy/
    │   │   │   ├── Powdery/
    │   │   │   └── Rust/
    │   ├── Train/
    │   │   ├── Train/
    │   │   │   ├── Healthy/
    │   │   │   ├── Powdery/
    │   │   │   └── Rust/
    │   └── Validation/
    │       └── Validation/
    │           ├── Healthy/
    │           ├── Powdery/
    │           └── Rust/
    ├── app.py
    ├── requirements.txt
    └── ...
    ```
*   **Automatic Download:** If the `data` directory (specifically the `Train` and `Validation` paths configured in `app.py`) is not found, and `kaggle.json` is present, the Streamlit app will attempt to download and extract the dataset into the `data` directory upon starting or when training is initiated.

## Usage

1.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    This will start the web application, typically opening it automatically in your default web browser.

2.  **Interact with the App:**
    *   The app will first attempt to load the pre-trained model (`plant_disease_model.pth`).
    *   Use the file uploader in the main panel to select a JPG or PNG image of a plant leaf.
    *   Click the "Classify Image" button.
    *   The predicted class (Healthy, Powdery, Rust) and the confidence score will be displayed.
    *   Optionally, probabilities for all classes are shown.

## Model

*   **Architecture:** ResNet-34.
*   **Weights:** Initialized with ImageNet pre-trained weights. The final fully connected layer is replaced to match the number of classes (3) in our dataset.
*   **Pre-trained File:** The trained model weights are saved in `plant_disease_model.pth`. The Streamlit app automatically loads this file on startup if it exists.

## Training

*   **Automatic Loading:** The application prioritizes loading the existing `plant_disease_model.pth`. Training is not required for basic prediction if this file is present and valid.
*   **Triggering Training:**
    *   If `plant_disease_model.pth` is not found, the app will indicate that training is needed.
    *   You can manually trigger training or retraining by clicking the "Train/Retrain Model" button in the sidebar of the Streamlit application.
*   **Process:**
    *   Training uses the images from the `data/Train/Train` directory and evaluates against the `data/Validation/Validation` directory.
    *   Hyperparameters (epochs, learning rate, optimizer, scheduler) are defined in the `train_config` dictionary within `app.py`. Default: 5 epochs, AdamW optimizer, ReduceLROnPlateau scheduler.
    *   Training progress (loss, accuracy per epoch) is displayed in the Streamlit app using a progress bar and logged to the console.
    *   Upon completion, the newly trained model weights are saved to `plant_disease_model.pth`, overwriting the previous file. The app will then use this updated model.
    *   Training history plots (Loss and Accuracy) may be displayed in the Streamlit app after training completes.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

*   Dataset provider: [Rashik Rahman Pritom on Kaggle](https://www.kaggle.com/rashikrahmanpritom)
*   Libraries used: PyTorch, Streamlit, Albumentations, etc.