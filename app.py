# -*- coding: utf-8 -*-
import streamlit as st
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import os
import random
from addict import Dict
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import time # For simulating progress if needed
import json
from zipfile import ZipFile

# --- Configuration ---

# Define paths relative to the app.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data") # Let's put data in a subfolder
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.pth")
KAGGLE_JSON_PATH = os.path.join(BASE_DIR, "kaggle.json")

config = Dict({
    "data_dir": DATA_DIR,
    "train_path": os.path.join(DATA_DIR, "Train", "Train"),
    "test_path": os.path.join(DATA_DIR, "Test", "Test"),
    "validation_path": os.path.join(DATA_DIR, "Validation", "Validation"),
    "model_save_path": MODEL_PATH,
    "kaggle_dataset": "rashikrahmanpritom/plant-disease-recognition-dataset"
})

train_config = Dict({
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "epochs": 5, # Keep epochs relatively low for web app training demo
    "seed": 2021,
    "image_shape": (128, 128),
    "image_channels": 3, # Assuming RGB
    "num_workers": 0, # Often best for Streamlit compatibility
    "batch_size": 32,

    "augmentations": A.Compose([
        A.Resize(height=128, width=128), # Ensure resize is always done
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Blur(p=0.1), # Keep augmentations light if needed
        ToTensorV2(), # Use ToTensorV2 for albumentations
    ]),
    "validation_augmentations": A.Compose([ # Simpler transforms for validation/test/prediction
        A.Resize(height=128, width=128),
        ToTensorV2(),
    ]),
    "optimizer": {
        "type": "AdamW",
        "parameters": {
            "lr": 0.001,
            "weight_decay": 0.01,
        }
    },
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "parameters": {
            "patience": 2,
            "mode": "min",
            "factor": 0.1,
        }
    }
})

# --- Utility Functions ---

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Comment out deterministic settings if they cause issues or slow down significantly
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False # Benchmark off for reproducibility if needed

def get_optimizer(model: torch.nn.Module, name: str = "SGD", parameters: dict = {}) -> torch.optim.Optimizer:
    optimizers = {
        "SGD": torch.optim.SGD,
        "AdamW": torch.optim.AdamW,
        "Adam": torch.optim.Adam,
        "RMSprop": torch.optim.RMSprop,
    }
    instance = optimizers.get(name, torch.optim.AdamW) # Default to AdamW
    optimizer = instance(model.parameters(), **parameters)
    return optimizer

def get_scheduler(optimizer: torch.optim.Optimizer, name: str, parameters: dict):
    schedulers = {
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "LambdaLR": torch.optim.lr_scheduler.LambdaLR,
        "StepLR": torch.optim.lr_scheduler.StepLR,
        # Add others if needed
    }
    instance = schedulers.get(name)
    if instance:
        scheduler = instance(optimizer, **parameters)
        return scheduler
    return None

def accuracy_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = predictions.argmax(dim=1) # Get class index from probabilities
    correct = (predictions == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy

# Logger setup (optional for Streamlit, prints to console)
def get_logger(name: str = __name__, format: str = "[%(asctime)s][%(levelname)s]: %(message)s") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers(): # Avoid adding multiple handlers on Streamlit reruns
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(format)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        # File handler might be less useful in deployed Streamlit
        # file_handler = logging.FileHandler(f"{name}.log")
        # file_handler.setLevel(logging.INFO)
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)
    logger.propagate = False
    return logger

logger = get_logger("plant_disease_app")

# --- Data Handling ---

def download_and_extract_data():
    """Downloads and extracts the dataset from Kaggle."""
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)
        logger.info("Data directory created.")

    zip_path = os.path.join(config.data_dir, "plant-disease-recognition-dataset.zip")

    # Check if data already extracted
    if os.path.exists(config.train_path) and os.path.exists(config.validation_path):
         logger.info("Dataset already extracted.")
         return True

    if not os.path.exists(zip_path):
        logger.info("Dataset zip file not found. Attempting to download from Kaggle...")
        if not os.path.exists(KAGGLE_JSON_PATH):
            st.error(f"Kaggle API credentials not found at {KAGGLE_JSON_PATH}. Cannot download dataset.")
            return False

        try:
            with open(KAGGLE_JSON_PATH) as f:
                kaggle_credentials = json.load(f)
            os.environ['KAGGLE_USERNAME'] = kaggle_credentials["username"]
            os.environ['KAGGLE_KEY'] = kaggle_credentials["key"]

            import kaggle
            kaggle.api.authenticate()
            st.info(f"Downloading dataset '{config.kaggle_dataset}'...")
            kaggle.api.dataset_download_files(config.kaggle_dataset, path=config.data_dir, unzip=False)
            st.info("Download complete.")
        except Exception as e:
            st.error(f"Failed to download dataset from Kaggle: {e}")
            logger.error(f"Kaggle download failed: {e}")
            return False

    if os.path.exists(zip_path):
        logger.info("Extracting dataset...")
        st.info("Extracting dataset...")
        try:
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(config.data_dir)
            logger.info("Extraction complete.")
            st.success("Dataset extracted successfully.")
            # Clean up zip file after extraction
            # os.remove(zip_path)
            # logger.info("Removed zip file.")
            return True
        except Exception as e:
            st.error(f"Failed to extract dataset: {e}")
            logger.error(f"Extraction failed: {e}")
            return False
    else:
        st.error("Dataset zip file not found after download attempt.")
        return False


class PlantDiseaseDataset(Dataset):
    def __init__(self, path, augmentations=None, image_shape=(256, 256), channels="RGB"):
        self.path = path
        self.image_shape = image_shape
        self.channels = channels
        self.augmentations = augmentations
        self.__images_labels = []
        self.labels = [] # Store labels in order

        if os.path.exists(path):
            # Sort labels alphabetically for consistency
            self.labels = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
            for label_idx, label in enumerate(self.labels):
                label_path = os.path.join(path, label)
                files = os.listdir(label_path)
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_path = os.path.join(label_path, file)
                        self.__images_labels.append((image_path, label_idx)) # Store index directly
        else:
            logger.warning(f"Dataset path not found: {path}")
            st.warning(f"Dataset path not found: {path}. Ensure data is downloaded and extracted.")

    def __len__(self):
        return len(self.__images_labels)

    def __getitem__(self, index):
        image_path, label_idx = self.__images_labels[index]

        # Load image using PIL
        try:
            image = Image.open(image_path).convert(self.channels)
            image_array = np.array(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a placeholder or skip? For now, return None and handle in collate_fn or loader
            # Let's try returning a black image as placeholder
            image_array = np.zeros((*self.image_shape, 3 if self.channels == "RGB" else 1), dtype=np.uint8)
            label_idx = 0 # Assign a default label maybe? Risky.


        # Apply augmentations if they exist
        if self.augmentations is not None:
            augmented = self.augmentations(image=image_array)
            image_tensor = augmented['image']
        else:
             # Basic resize and ToTensor if no augmentations
            basic_transform = A.Compose([
                A.Resize(height=self.image_shape[0], width=self.image_shape[1]),
                ToTensorV2()
            ])
            image_tensor = basic_transform(image=image_array)['image']

        # Normalize tensor to [0, 1] if ToTensorV2 didn't already
        if image_tensor.max() > 1.0:
             image_tensor = image_tensor / 255.0

        # Ensure correct type
        image_tensor = image_tensor.float()


        return {
            "image": image_tensor,
            "label": torch.tensor(label_idx, dtype=torch.long), # Use Long for CrossEntropyLoss
        }

# Simple collate function (adjust if getitem can return None)
def collate_fn(batch):
    # Filter out None items if any image failed to load in __getitem__
    batch = [item for item in batch if item is not None]
    if not batch:
        return {"images": torch.empty(0), "labels": torch.empty(0)}

    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return {
        "images": images,
        "labels": labels
    }

# --- Model Definition ---

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=3): # Default to 3 classes found in dataset
        super().__init__()
        # Use a lighter model like ResNet18 or MobileNetV2 for faster training/inference if needed
        self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1) # Use updated weights API

        # Freeze all parameters first
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        # Note: CrossEntropyLoss in PyTorch combines LogSoftmax and NLLLoss,
        # so you usually DON'T need a Softmax layer here if using CrossEntropyLoss.
        # If you need probabilities, apply softmax *after* the model output during inference.

    def forward(self, image):
        return self.model(image)

# --- Training Logic ---

class Trainer:
    def __init__(self, model, criterion, optimizer, metric, scheduler=None, logger=None, device="cpu", progress_bar=None):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.device = device
        self.metric = metric # Assumes metric takes (predictions, targets)
        self.history = Dict({'train_losses': [], 'train_scores': [], 'val_losses': [], 'val_scores': []})
        self.progress_bar = progress_bar # Streamlit progress bar

    def _update_progress(self, epoch, total_epochs, batch_idx, total_batches, loss, score, phase="Train"):
        if self.progress_bar:
            progress = (epoch * total_batches + batch_idx + 1) / (total_epochs * total_batches)
            self.progress_bar.progress(min(progress, 1.0),
                                       text=f"Epoch {epoch+1}/{total_epochs} [{phase}] Batch {batch_idx+1}/{total_batches} | Loss: {loss:.4f} | Acc: {score:.4f}")

    def evaluate(self, loader, epoch=0, total_epochs=1):
        self.model.eval()
        loss, score, count = 0.0, 0.0, 0
        total_batches = len(loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                images = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device) # Move labels to device too

                outputs = self.model(images)
                batch_loss = self.criterion(outputs, labels)
                batch_score = self.metric(outputs.cpu(), labels.cpu()) # Metric might run on CPU

                loss += batch_loss.item() * images.size(0)
                score += batch_score * images.size(0)
                count += images.size(0)

                # Update progress bar during validation too
                self._update_progress(epoch, total_epochs, batch_idx, total_batches, batch_loss.item(), batch_score, phase="Valid")


        avg_loss = loss / count
        avg_score = score / count
        return avg_loss, avg_score

    def fit(self, train_loader, validation_loader=None, epochs=10):
        train_length = len(train_loader)
        if validation_loader:
            val_length = len(validation_loader)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss, epoch_score, train_count = 0.0, 0.0, 0

            for batch_idx, batch in enumerate(train_loader):
                images = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Calculate metrics
                batch_score = self.metric(outputs.cpu(), labels.cpu()) # Metric on CPU
                epoch_loss += loss.item() * images.size(0)
                epoch_score += batch_score * images.size(0)
                train_count += images.size(0)

                # Update progress bar
                self._update_progress(epoch, epochs, batch_idx, train_length, loss.item(), batch_score, phase="Train")

            avg_epoch_loss = epoch_loss / train_count
            avg_epoch_score = epoch_score / train_count
            self.history.train_losses.append(avg_epoch_loss)
            self.history.train_scores.append(avg_epoch_score)
            log_msg = f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_epoch_loss:.4f} | Train Acc: {avg_epoch_score:.4f}"

            # Validation Step
            if validation_loader:
                val_loss, val_score = self.evaluate(validation_loader, epoch=epoch, total_epochs=epochs)
                self.history.val_losses.append(val_loss)
                self.history.val_scores.append(val_score)
                log_msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_score:.4f}"

                # Scheduler Step (based on validation loss)
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step() # For schedulers not needing a metric

            if self.logger: self.logger.info(log_msg)

            # Update progress bar text at end of epoch
            if self.progress_bar:
                self.progress_bar.progress( (epoch+1)/epochs, text=f"Epoch {epoch+1}/{epochs} Complete. {log_msg}")


        # Clear progress bar at the end
        if self.progress_bar:
            time.sleep(1) # Give time to see final message
            self.progress_bar.empty()

        return self.history

# --- Streamlit UI ---

st.set_page_config(page_title="Plant Disease Classifier", layout="wide")
st.title("🌿 Plant Disease Classification")
st.subheader("By Taufeeq Riyaz - 1RVU23CSE506")

# Global variable to hold class names (populated once)
CLASS_NAMES = []

# --- Model Loading ---
@st.cache_resource # Cache the model resource
def load_pytorch_model(model_path, num_classes):
    """Loads the PyTorch model state dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantDiseaseModel(num_classes=num_classes)
    if os.path.exists(model_path):
        try:
            # Load state dict, handling potential device mismatches
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval() # Set to evaluation mode
            logger.info(f"Model loaded successfully from {model_path} onto {device}")
            return model, device
        except Exception as e:
            st.error(f"Error loading model: {e}. Consider retraining.")
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None, device # Return None if loading fails
    else:
        logger.warning(f"Model file not found at {model_path}. Need to train first.")
        return None, device # Return None if file doesn't exist

def get_class_names(data_path):
    """Gets class names from the directory structure."""
    if not os.path.exists(data_path):
        return []
    return sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

# --- Prediction Function ---
def predict(model, image_bytes, device, class_names):
    """Preprocesses image bytes and makes a prediction."""
    transform = train_config.validation_augmentations # Use validation transforms for prediction
    try:
        image = Image.open(image_bytes).convert("RGB")
        image_array = np.array(image)
        transformed = transform(image=image_array)
        image_tensor = transformed['image'].unsqueeze(0).to(device) # Add batch dim and send to device

         # Normalize if needed (ToTensorV2 usually handles this)
        if image_tensor.max() > 1.0:
             image_tensor = image_tensor / 255.0
        image_tensor = image_tensor.float()


        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0] # Apply softmax here for probabilities
            predicted_idx = torch.argmax(probabilities).item()
            predicted_class = class_names[predicted_idx]
            confidence = probabilities[predicted_idx].item()
            return predicted_class, confidence, probabilities.cpu().numpy()

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        logger.error(f"Prediction error: {e}")
        return None, None, None

# --- Training Function ---
def run_training(progress_widget):
    """Runs the full training process."""
    st.info("Starting training process...")
    seed_everything(train_config.seed)

    # 1. Prepare Data
    st.info("Checking and preparing dataset...")
    if not download_and_extract_data():
        st.error("Failed to get dataset. Training cannot proceed.")
        return False, None # Indicate failure and no history

    global CLASS_NAMES
    CLASS_NAMES = get_class_names(config.train_path)
    if not CLASS_NAMES:
         st.error("Could not determine class names from the training directory.")
         return False, None
    num_classes = len(CLASS_NAMES)
    st.info(f"Found {num_classes} classes: {', '.join(CLASS_NAMES)}")


    train_dataset = PlantDiseaseDataset(path=config.train_path,
                                        augmentations=train_config.augmentations,
                                        image_shape=train_config.image_shape)
    validation_dataset = PlantDiseaseDataset(path=config.validation_path,
                                             augmentations=train_config.validation_augmentations, # Use validation augs
                                             image_shape=train_config.image_shape)

    if not train_dataset or not validation_dataset or len(train_dataset) == 0:
        st.error("Failed to load datasets. Check data paths and integrity.")
        return False, None

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=train_config.batch_size,
                              num_workers=train_config.num_workers,
                              shuffle=True,
                              collate_fn=collate_fn,
                              pin_memory=True) # Pin memory if using GPU
    validation_loader = DataLoader(dataset=validation_dataset,
                                   batch_size=train_config.batch_size * 2, # Larger batch size for validation
                                   num_workers=train_config.num_workers,
                                   shuffle=False,
                                   collate_fn=collate_fn,
                                   pin_memory=True)

    # 2. Initialize Model, Optimizer, Criterion, Scheduler
    st.info("Initializing model and training components...")
    model = PlantDiseaseModel(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, train_config.optimizer.type, train_config.optimizer.parameters)
    scheduler = get_scheduler(optimizer, train_config.scheduler.type, train_config.scheduler.parameters) if "scheduler" in train_config else None

    # 3. Setup Trainer
    trainer = Trainer(model=model,
                      criterion=criterion,
                      optimizer=optimizer,
                      metric=accuracy_score, # Pass the function directly
                      scheduler=scheduler,
                      logger=logger,
                      device=train_config.device,
                      progress_bar=progress_widget) # Pass the progress bar widget

    # 4. Start Training
    st.info(f"Starting training for {train_config.epochs} epochs on {train_config.device}...")
    history = trainer.fit(train_loader=train_loader,
                          validation_loader=validation_loader,
                          epochs=train_config.epochs)

    # 5. Save Model
    st.info("Training finished. Saving model...")
    try:
        torch.save(model.state_dict(), config.model_save_path)
        st.success(f"Model saved successfully to {config.model_save_path}")
        # Clear cache for the loaded model so it reloads the new one
        load_pytorch_model.clear()
        return True, history # Indicate success and return history
    except Exception as e:
        st.error(f"Error saving model: {e}")
        logger.error(f"Failed to save model: {e}")
        return False, history # Indicate failure but return history


# --- Main App Logic ---

# Check if data exists, prompt for download/extraction if needed
data_ready = download_and_extract_data()

# Try to get class names (needed early for model loading)
if data_ready:
    CLASS_NAMES = get_class_names(config.train_path)
    if not CLASS_NAMES:
        st.warning("Could not determine class names. Retraining might be necessary.")
        num_classes = 3 # Default or guess
    else:
        num_classes = len(CLASS_NAMES)
else:
    st.warning("Dataset not found or couldn't be prepared. Prediction and training might fail.")
    num_classes = 3 # Default

# Attempt to load the model
model, device = load_pytorch_model(config.model_save_path, num_classes)

# --- Sidebar for Options ---
st.sidebar.title("Options")

# Retraining Button
if st.sidebar.button("Train/Retrain Model"):
    if not data_ready:
        st.sidebar.error("Dataset is not available. Cannot train.")
    else:
        # Clear the model cache before retraining
        load_pytorch_model.clear()
        model = None # Ensure model is considered not loaded

        st.sidebar.info("Model training requested.")
        progress_bar = st.progress(0, text="Initializing training...")
        training_success, history = run_training(progress_bar)

        if training_success:
            st.sidebar.success("Training complete! Model updated.")
            # Attempt to reload the newly trained model
            model, device = load_pytorch_model(config.model_save_path, len(CLASS_NAMES))
            if model:
                 st.info("New model loaded successfully for prediction.")
            else:
                 st.error("Failed to load the newly trained model.")

            # Optionally display training curves
            if history:
                st.subheader("Training History")
                epochs_range = range(1, len(history.train_losses) + 1)
                fig, ax = plt.subplots(1, 2, figsize=(15, 5))

                # Loss Plot
                ax[0].plot(epochs_range, history.train_losses, 'r-o', label='Train Loss')
                if history.val_losses:
                    ax[0].plot(epochs_range, history.val_losses, 'b-o', label='Validation Loss')
                ax[0].set_title('Loss per Epoch')
                ax[0].set_xlabel('Epochs')
                ax[0].set_ylabel('Loss')
                ax[0].legend()
                ax[0].grid(True)

                # Accuracy Plot
                ax[1].plot(epochs_range, history.train_scores, 'r-o', label='Train Accuracy')
                if history.val_scores:
                    ax[1].plot(epochs_range, history.val_scores, 'b-o', label='Validation Accuracy')
                ax[1].set_title('Accuracy per Epoch')
                ax[1].set_xlabel('Epochs')
                ax[1].set_ylabel('Accuracy')
                ax[1].legend()
                ax[1].grid(True)

                st.pyplot(fig)

        else:
            st.sidebar.error("Training process failed or was incomplete.")

# --- Main Area for Prediction ---
st.header("Classify an Image")

if model is None:
    st.warning("⚠️ Model not loaded. Please train the model using the sidebar option.")
else:
    st.success(f"✅ Model loaded successfully ({os.path.basename(config.model_save_path)})")
    if not CLASS_NAMES:
         st.warning("Class names not determined. Prediction output might be just an index.")

    uploaded_file = st.file_uploader("Choose an image file (jpg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=False)

        # Predict button
        if st.button('Classify Image'):
            if not CLASS_NAMES:
                 st.error("Cannot classify: Class names are missing.")
            else:
                with st.spinner('Classifying...'):
                    predicted_class, confidence, probabilities = predict(model, uploaded_file, device, CLASS_NAMES)

                if predicted_class is not None:
                    st.subheader(f"Prediction: {predicted_class}")
                    st.write(f"Confidence: {confidence:.2%}")

                    # Optional: Display probabilities for all classes
                    st.subheader("Class Probabilities:")
                    prob_data = { "Class": CLASS_NAMES, "Probability": [f"{p:.2%}" for p in probabilities]}
                    st.dataframe(prob_data)
                else:
                    st.error("Could not classify the image.")