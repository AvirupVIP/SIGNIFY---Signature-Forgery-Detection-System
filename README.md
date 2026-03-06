# SIGNIFY — Signature Forgery Detection System

Signify is a digital signature verification web application that provides intelligent authentication to ensure document integrity and eliminate fraudulent signature risks. It combines a responsive frontend, a Flask-based REST API backend, and machine learning / deep learning models for signature analysis.

## Live Demo

**Frontend (GitHub Pages):** https://avirupvip.github.io/SIGNIFY---Signature-Forgery-Detection-System/

**Backend API (Render):** https://signature-e63s.onrender.com

## Project Structure

```
SIGNIFY/
├── index.html                      # Main frontend page (EmailJS contact form, popup modals)
├── .gitignore
├── README.md
│
├── assets/                         # Frontend assets
│   ├── css/
│   │   ├── home.css                # Main stylesheet
│   │   ├── popup.css               # Modal/popup styles
│   │   └── animation.css           # Magnifier animation styles
│   ├── js/
│   │   ├── home.js                 # Theme toggle, scroll, EmailJS contact form
│   │   ├── popup.js                # Signature upload & verification modals
│   │   └── animation.js            # Magnifier scanning animation
│   └── img/
│       ├── Signify_logo.png
│       ├── About.png
│       └── Animation_signature.svg
│
├── backend/                        # Flask REST API
│   ├── app.py                      # API server (endpoints: /add-customer, /verify-signature, /customer/<id>)
│   ├── preprocessing.py            # Image preprocessing pipeline
│   ├── feature_extraction.py       # HOG + multi-feature extraction & cosine similarity
│   ├── requirements.txt            # Python dependencies
│   ├── Procfile                    # Render deployment config
│   └── runtime.txt                 # Python version (3.11.9)
│
└── ml_models/                      # Machine learning research & models
    ├── ayan/                       # Siamese network model & training data
    │   ├── d.ipynb                 # Training notebook
    │   ├── siamese1.ipynb          # Siamese architecture notebook
    │   ├── signature_siamese_model.h5  # Trained Siamese model weights
    │   ├── forg.jpeg               # Sample forged signature
    │   ├── genuine.jpeg            # Sample genuine signature
    │   ├── more forg.jpeg          # Additional forged sample
    │   ├── processed/              # Processed signature images
    │   └── processed_signature/    # Full signature dataset (train/test splits)
    │
    └── notebooks/                  # MobileNet & Siamese on Big_AHH_Dataset
        ├── Big_AHH_Dataset.ipynb   # Main dataset notebook
        ├── MobileNet_V1.ipynb      # MobileNet V1 implementation
        ├── MobileNet_V2.ipynb      # MobileNet V2 implementation
        ├── MobileNet_V3(_another_larger_dataset).ipynb
        ├── MobileNet_V4.ipynb      # Latest MobileNet implementation
        ├── siamese1.ipynb          # Siamese network implementation
        ├── tripletloss.ipynb       # Triplet loss implementation
        ├── preprocessing.ipynb     # Data preprocessing notebook
        ├── sig_forg2.ipynb         # Signature forgery analysis
        ├── signature_siamese_model.h5  # Trained model weights
        ├── Extra Processing/       # Additional processing notebooks & data
        ├── ExtraProcessing2/       # Extended processing pipeline
        └── processed_signature/    # Full processed signature dataset
```

## Features

- **Add Customer**: Register a customer with 6–12 signature samples for training
- **Verify Signature**: Upload a signature and compare against stored samples using cosine similarity
- **Customer Lookup**: Real-time customer ID existence check
- **EmailJS Contact Form**: Send messages directly from the frontend
- **Theme Toggle**: Light/dark mode support
- **Responsive Design**: Mobile-friendly layout with hamburger menu

## Setup

### Frontend

Open `index.html` in a browser or serve via a local HTTP server (e.g., VS Code Live Server on port 5500).

### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The backend API runs on `http://localhost:5000` by default. For production, the frontend uses the Render deployment URL configured in `assets/js/popup.js` (`BACKEND_URL` constant).

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/add-customer` | Register customer with signature images |
| `POST` | `/verify-signature` | Verify a signature against stored samples |
| `GET` | `/customer/<id>` | Check if a customer exists |

#### Deployment (Render)

The `backend/Procfile` configures Render deployment:
```
web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
```
Run the deployment from the `backend/` directory.

### ML Models

The `ml_models/` directory contains Jupyter notebooks and trained models for research:

```bash
# Install Jupyter and dependencies
pip install jupyter tensorflow scikit-learn scikit-image opencv-python numpy

# Open notebooks
jupyter notebook ml_models/
```

- **`ml_models/ayan/`** — Siamese network approach with processed signature dataset
- **`ml_models/notebooks/`** — MobileNet V1–V4 and Siamese network trained on the Big_AHH_Dataset

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, JavaScript |
| Backend | Python 3.11, Flask, Flask-CORS |
| Image Processing | OpenCV, scikit-image, NumPy |
| Feature Extraction | HOG (Histogram of Oriented Gradients) + cosine similarity |
| ML/DL Models | TensorFlow/Keras, MobileNet, Siamese Networks |
| Deployment | GitHub Pages (frontend), Render (backend) |
| Email | EmailJS |

## Contributors

- **AvirupVIP** — Project lead, frontend & backend integration
- **messiayan** — Siamese network model & training data (`ml_models/ayan/`)
- **spiralarchitect5** — MobileNet & Siamese on Big_AHH_Dataset (`ml_models/notebooks/`)
- **pratikn2003** — Backend API development

