# Automated Weather Classification using Transfer Learning

A deep learning project that classifies weather conditions from images using transfer learning on pre-trained CNN architectures. Four models are benchmarked (VGG16, ResNet50, InceptionV3, Xception), and the best-performing model is deployed as a Flask web app for real-time predictions.

## Overview

Instead of training a CNN from scratch, this project leverages ImageNet-pretrained backbones and fine-tunes a classification head on weather images. The convolutional layers are frozen to retain the rich visual features learned during pre-training, and a new softmax classifier is trained on top to predict one of four weather classes:

- **Cloudy**
- **Rain**
- **Shine**
- **Sunrise**

## Dataset

- **Source:** Multi-class Weather Dataset
- **Total images:** 1,125
- **Training set:** 901 images
- **Validation set:** 224 images
- **Split:** 80% training / 20% validation
- **Input size:** 224 × 224 × 3

Data augmentation applied during training: rescaling (1/255), shear, zoom, and horizontal flipping.

## Models Compared

| Model | Loss | Accuracy | Val Loss | Val Accuracy |
|--------------|-------|----------|----------|--------------|
| **VGG16**    | 0.052 | **0.99** | 0.30     | **0.89**     |
| ResNet50     | 1.12  | 0.53     | 1.13     | 0.48         |
| InceptionV3  | 0.14  | 0.95     | 0.53     | 0.88         |
| Xception     | 0.12  | 0.96     | 0.62     | 0.87         |

**VGG16** delivered the best validation accuracy and is used for deployment.

## Tech Stack

- **Language:** Python 3
- **Deep Learning:** TensorFlow, Keras
- **Web Framework:** Flask
- **Frontend:** HTML, CSS, JavaScript (jQuery), Bootstrap 4
- **Image Processing:** Pillow, NumPy
- **Notebook:** Google Colab

## Project Structure

```
Automated_weather_classification_using_transfer_learning/
│
├── app.py                                       # Flask web application
├── Automated_weather_classification.h5          # Trained VGG16 model
├── AUTOMATED_WEATHER_CLASSIFICATION_USING_TRANSFER_LEARNING.ipynb  # Training notebook
│
├── static/
│   ├── css/
│   │   └── main.css                             # Styling for the UI
│   └── js/
│       └── main.js                              # Image preview + AJAX prediction
│
├── templates/
│   ├── index.html                               # Landing page (Home / About / Images / Predict)
│   ├── input.html                               # Simple upload form
│   └── output.html                              # Renders prediction result
│
├── uploads/                                     # User-uploaded images (runtime)
│   ├── cloudy/                                  # Sample test images per class
│   ├── rain/
│   ├── shine/
│   └── sunrise/
│
└── README.md
```

## How It Works

1. **Load pre-trained VGG16** with ImageNet weights, excluding the top classification layers.
2. **Freeze** all convolutional layers to preserve learned feature representations.
3. **Add a custom head:** Flatten layer followed by a Dense softmax layer with 4 output classes.
4. **Train** the model with Adam optimizer and categorical cross-entropy loss for 10 epochs.
5. **Save** the trained model as `Automated_weather_classification.h5`.
6. **Deploy** via Flask: the user uploads an image through the web UI, the image is preprocessed (resized to 224×224 and passed through VGG19 `preprocess_input`), the model predicts, and the result is displayed via AJAX without a page reload.

## Web Interface

The Flask app exposes the following routes:

| Route       | Method   | Description                                     |
|-------------|----------|-------------------------------------------------|
| `/`         | GET      | Renders `index.html` (main landing page)        |
| `/home`     | GET      | Alias for the home page                         |
| `/input`    | GET      | Renders the minimal upload form (`input.html`)  |
| `/predict`  | GET/POST | Accepts an image, runs inference, returns class |

The main page is organized into four sections: **Home**, **About**, **Images**, and **Predict** — accessible via the top nav bar.

## Setup & Run Locally

### Prerequisites

```bash
pip install flask tensorflow numpy pillow
```

### Steps

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/Automated_weather_classification_using_transfer_learning.git
   cd Automated_weather_classification_using_transfer_learning
   ```

2. Make sure `Automated_weather_classification.h5` is in the project root.

3. Create the `uploads/` folder if it doesn't exist:
   ```bash
   mkdir uploads
   ```

4. Run the Flask app:
   ```bash
   python app.py
   ```

5. Open your browser at `http://127.0.0.1:5000/`, upload a weather image, and view the prediction.

## Training (Optional)

To retrain from scratch, open the Colab notebook and:
1. Upload the dataset archive to your Google Drive.
2. Update the dataset path in the notebook.
3. Run all cells. The final cell saves the model as `.h5`.

## Results

VGG16 reached **99% training accuracy** and **89% validation accuracy** after 10 epochs, significantly outperforming ResNet50 in this small-dataset setting. The Flask interface correctly classifies sample images of sunrises, shine, rain, and cloudy weather (sample images included under `uploads/`).

## Applications

- Weather forecasting support systems
- Environmental and climate monitoring
- Aviation and transportation planning
- Agriculture and irrigation scheduling
- Solar and wind energy forecasting

## Future Work

- Train on a larger, weather-specific dataset with more classes (snow, fog, haze, storms)
- Explore lightweight architectures (MobileNet, EfficientNet) for edge deployment
- Add Grad-CAM visualizations for explainability
- Integrate multi-modal inputs (satellite imagery + sensor readings)
- Deploy to a cloud platform (Heroku, Render, AWS)

## Author

**K Sarath** (Reg No: 20BCI0171)
VIT Vellore — Externship Project
Guide: Dr. Sharmila Banu

## References

1. M. Abrar et al., *Weather Prediction using Classification*, Science International, 2014.
2. A. John Arnfield, *Climate Classification*, Encyclopedia Britannica, 2016.
3. J. Hidalgo, R. Jougla, *On the use of local weather types classification to improve climate understanding*, PLoS ONE, 2018.
4. H. Beck et al., *Present and future Köppen-Geiger climate classification maps at 1-km resolution*, Sci Data, 2018.
