# Music Classifier and Recommender System

## Overview

This repository contains the implementation of a **Music Genre Classifier and Recommender System** using machine learning techniques. The project uses audio feature extraction, classification algorithms (like SVM), and recommendation methods to classify music and recommend similar songs based on user-uploaded files.

The system leverages various preprocessing steps, including **Label Encoding**, **Standard Scaling**, and **Principal Component Analysis (PCA)** to process and reduce the dimensionality of audio features. The classifier predicts the genre of the uploaded song, while the recommender suggests songs similar to the uploaded song based on cosine similarity.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python 3.x** is installed on your machine.
- **Git** for version control.
- **Virtual Environment (Optional but recommended)** for managing dependencies.

### Step-by-step Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/music-classifier-recommender.git](https://github.com/Just-NK14/music-classifier-recommender.git)
   ```

2. Navigate to the project directory:
   ```bash
   cd music-classifier-recommender
   ```

3. (Optional) Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   venv\Scripts\activate     # For Windows
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the Music Genre Classifier and Recommender System, follow these steps:

1. **Run the Streamlit app**:
   Navigate to the `src` folder and run the Streamlit app:

   ```bash
   cd src
   streamlit run app.py
   ```

2. **Upload an audio file**:
   - Once the app is running, you can upload an audio file (in `.wav` format) for classification and recommendation.
   - The app will display the predicted genre and suggest similar songs based on the uploaded file.

## Features

- **Music Genre Classification**: Classifies uploaded audio files into predefined music genres.
- **Song Recommendation**: Recommends similar songs to the uploaded file based on audio features and cosine similarity.
- **Streamlit Interface**: A user-friendly web interface to interact with the system.
- **Feature Extraction**: Extracts relevant audio features (such as MFCCs) for classification and recommendation.
- **Dimensionality Reduction**: Uses PCA for reducing the dimensionality of audio features to improve model performance.

## Project Structure

The project has the following structure:

```
music-classifier-recommender/
│
├── data/                            # Folder containing raw and processed data
│   ├── raw/                         # Raw audio files and genre data
│   └── processed/                   # Processed data files (e.g., CSV, pickled models)
│
├── notebooks/                       # Jupyter notebooks for experimentation
│
├── src/                             # Source code for the Streamlit app
│   ├── app.py                       # Streamlit app script
│   └── ...                          # Other Python modules
│
├── requirements.txt                 # List of dependencies
├── .gitignore                       # Git ignore file
└── README.md                        # This file
```

## Contributing

We welcome contributions to the project! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add feature'`).
5. Push to your forked repository (`git push origin feature-branch`).
6. Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Customizations:

- Modify the **Features** section if you have more specific capabilities to highlight.
- Update the **Usage** section based on the exact instructions or options you provide to users (like model parameters, etc.).
- Adjust the **Project Structure** based on your actual folder structure and files.
