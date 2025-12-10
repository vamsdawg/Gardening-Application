import json
import os

filepath = r"g:\DSBA\DSBA 6211\Project\Gardening-Application\models\train_weed_detector.ipynb"

content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Train YOLOv8 Weed Detector\n",
    "\n",
    "This notebook guides you through training a custom YOLOv8 model to detect weeds using the [CropWeeds-YOLO Dataset](https://www.kaggle.com/datasets/swish9/weeds-detection).\n",
    "\n",
    "The resulting model can be used in the `LawnAnalyzer` class to improve weed detection accuracy."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Install Dependencies\n",
    "\n",
    "Ensure `ultralytics` is installed. We also need `kaggle` to download the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%pip install ultralytics kaggle"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Download Dataset from Kaggle\n",
    "\n",
    "**Prerequisite:** You need a `kaggle.json` API token.\n",
    "1. Go to your Kaggle Account settings.\n",
    "2. Click \"Create New API Token\".\n",
    "3. Place the `kaggle.json` file in `C:\\Users\\<YourUser>\\.kaggle\\` (Windows) or `~/.kaggle/` (Linux/Mac).\n",
    "\n",
    "Alternatively, you can manually download the dataset from [here](https://www.kaggle.com/datasets/swish9/weeds-detection), unzip it, and place it in a folder named `datasets/weeds-detection` in the project root.\n",
    "\n",
    "The code below attempts to download it automatically."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "# Define paths\n",
    "project_root = Path(\"..\").resolve() # Assuming this notebook is in models/\n",
    "dataset_dir = project_root / \"datasets\" / \"weeds-detection\"\n",
    "\n",
    "# Create datasets directory if it doesn't exist\n",
    "dataset_dir.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "# Download dataset using Kaggle API\n",
    "# Note: This requires kaggle.json to be set up correctly\n",
    "try:\n",
    "    import kaggle\n",
    "    print(\"Downloading dataset...\")\n",
    "    kaggle.api.dataset_download_files('swish9/weeds-detection', path=dataset_dir, unzip=True)\n",
    "    print(\"Download complete.\")\n",
    "except Exception as e:\n",
    "    print(f\"Could not download automatically: {e}\")\n",
    "    print(f\"Please manually download the dataset to: {dataset_dir}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Prepare Dataset Structure\n",
    "\n",
    "The dataset should already be in YOLO format. Let's verify the structure.\n",
    "It typically contains `train`, `test`, and `val` folders, each with `images` and `labels`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Verify structure\n",
    "print(f\"Checking contents of {dataset_dir}...\")\n",
    "for item in dataset_dir.iterdir():\n",
    "    print(item.name)\n",
    "\n",
    "# Define paths for data.yaml\n",
    "train_path = dataset_dir / \"train\" / \"images\"\n",
    "val_path = dataset_dir / \"val\" / \"images\"\n",
    "\n",
    "if not train_path.exists():\n",
    "    print(f\"Warning: Train path {train_path} does not exist. Check the unzipped structure.\")\n",
    "else:\n",
    "    print(f\"Train path verified: {train_path}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Create YOLO Configuration (YAML)\n",
    "\n",
    "We need to create a `data.yaml` file that tells YOLO where the images are and what the classes are.\n",
    "Based on the dataset description, it detects weeds and crops. We need to check the `data.yaml` if it came with one, or create our own."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import yaml\n",
    "\n",
    "# Define the data configuration\n",
    "# Note: You might need to adjust class names based on the specific dataset metadata if available.\n",
    "# Assuming standard classes for this dataset (0: crop, 1: weed) - verify this!\n",
    "data_config = {\n",
    "    'path': str(dataset_dir.absolute()),\n",
    "    'train': 'train/images',\n",
    "    'val': 'val/images',\n",
    "    'test': 'test/images',\n",
    "    'nc': 2,\n",
    "    'names': ['crop', 'weed'] \n",
    "}\n",
    "\n",
    "yaml_path = dataset_dir / \"data.yaml\"\n",
    "\n",
    "with open(yaml_path, 'w') as f:\n",
    "    yaml.dump(data_config, f)\n",
    "\n",
    "print(f\"Created configuration at {yaml_path}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Initialize YOLOv8 Model\n",
    "\n",
    "We will use `yolov8n.pt` (Nano) as the starting point. It's small and fast, suitable for the application."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from ultralytics import YOLO\n",
    "\n",
    "# Load a model\n",
    "model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Train the Model\n",
    "\n",
    "Train the model for a specified number of epochs. 50-100 is usually a good start."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train the model\n",
    "results = model.train(\n",
    "    data=str(yaml_path),\n",
    "    epochs=50,\n",
    "    imgsz=640,\n",
    "    patience=10,\n",
    "    batch=16,\n",
    "    name='yolov8n_weeds'\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Evaluate Model Performance\n",
    "\n",
    "Check the validation metrics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Validate the model\n",
    "metrics = model.val()\n",
    "print(f\"mAP50-95: {metrics.box.map}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Export Model\n",
    "\n",
    "Save the best model to the `models/` directory so the app can use it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import shutil\n",
    "\n",
    "# The best model is saved in runs/detect/yolov8n_weeds/weights/best.pt\n",
    "best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'\n",
    "target_path = project_root / 'models' / 'weed_detector.pt'\n",
    "\n",
    "if best_model_path.exists():\n",
    "    shutil.copy(best_model_path, target_path)\n",
    "    print(f\"Model saved to {target_path}\")\n",
    "    print(\"Update your LawnAnalyzer class to load this model: self.yolo_model = YOLO('models/weed_detector.pt')\")\n",
    "else:\n",
    "    print(\"Could not find best.pt\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.x"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open(filepath, 'w') as f:
    json.dump(content, f, indent=1)
