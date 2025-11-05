# Data Types & Preprocessing for ML / Analytics - Complete Guide

## 📋 Table of Contents
1. [Numeric Data](#-handling-numeric-data)
2. [Categorical Data](#-handling-categorical-data)
3. [Boolean Data](#-handling-boolean-data)
4. [Complex/Structured Data](#-handling-complex--structured-data)
5. [Derived/Engineered Features](#-handling-derived--engineered-features)
6. [Text/String Data](#-handling-text--string-data-nlp-preprocessing)
7. [Geospatial & Identifier Data](#-handling-geospatial--identifier-data)
8. [Multimedia Data](#-handling-multimedia-data-images-audio-video)

---

## 📘 Handling Numeric Data

### 1️⃣ Understanding Numeric Data
- Continuous or discrete values representing measurable quantities
- Examples: Age, Salary, Temperature, Height, Weight

### 2️⃣ Common Preprocessing Steps
- **Scaling Methods**:
  - Standardization (mean=0, std=1)
  - Normalization (min=0, max=1)
- **Handling Outliers**:
  - IQR method
  - Z-score method
  - Winsorization
- **Transformations**:
  - Log transformation
  - Square root transformation
  - Box-Cox transformation

### 3️⃣ Notes for ML/Analytics
- Required for distance-based models (KNN, SVM, K-means)
- Tree-based models (Random Forest, XGBoost) are less sensitive to scaling
- Can derive features: ratios, percentages, aggregates, rolling statistics

---

## 📘 Handling Categorical Data

### 1️⃣ Understanding Categorical Data
- Represents labels or categories with limited possible values
- **Nominal**: No order (Gender, Country, Color)
- **Ordinal**: Has order (Education Level, Rating, Size)

### 2️⃣ Common Preprocessing Steps
- **Label Encoding**: Convert to integers (for ordinal data)
- **One-Hot Encoding**: Create binary columns for each category (for nominal data)
- **Frequency Encoding**: Replace categories with their frequency counts
- **Target Encoding**: Replace with mean of target variable for each category
- **Binary Encoding**: Combine one-hot and binary encoding

### 3️⃣ Notes for ML/Analytics
- High-cardinality features may need dimensionality reduction
- One-hot encoding can lead to curse of dimensionality
- Tree-based models can handle label encoding well
- Linear models require one-hot encoding

---

## 📘 Handling Boolean Data

### 1️⃣ Understanding Boolean Data
- Binary values representing True/False conditions
- Examples: IsActive, HasAccount, IsPremiumUser

### 2️⃣ Preprocessing Steps
- Convert to integer (0 for False, 1 for True)
- Ensure consistent mapping across dataset
- Handle missing values appropriately

### 3️⃣ Notes for ML/Analytics
- Can be used directly in most ML models
- Useful as flags and conditional features
- Important for filtering and segmentation in analytics

---

## 📘 Handling Complex / Structured Data

### 1️⃣ Understanding Structured Data
- Nested or hierarchical formats
- Examples: JSON, XML, time-series, graphs, arrays

### 2️⃣ Preprocessing Steps
- **Flattening**: Convert nested structures to flat tables
- **Time-series**: Extract temporal features (hour, day, month, season)
- **Graph Data**: Node features, edge weights, graph embeddings
- **Array Data**: Reshape, normalize, extract statistical features

### 3️⃣ Notes for ML/Analytics
- Requires domain-specific feature extraction
- Time-series: autocorrelation, rolling windows, seasonality
- Graph data: network metrics, community detection
- Often used in deep learning models

---

## 📘 Handling Derived / Engineered Features

### 1️⃣ Understanding Derived Data
- Features created from existing variables to enhance predictive power
- Domain knowledge-driven transformations

### 2️⃣ Examples of Derived Features
- **Mathematical**: ratios, differences, products, polynomials
- **Temporal**: time since last event, moving averages
- **Interaction Terms**: combinations of features
- **Aggregations**: group-level statistics
- **Domain-specific**: BMI from height/weight, revenue per customer

### 3️⃣ Notes for ML/Analytics
- Can significantly improve model performance
- Balance between complexity and interpretability
- Avoid data leakage in time-based features
- Validate feature importance

---

## 📘 Handling Text / String Data (NLP Preprocessing)

### 1️⃣ Understanding Text Data
- Unstructured sequence of characters (words, sentences, paragraphs)
- Examples: reviews, emails, social media posts, documents
- Challenges: punctuation, stopwords, slang, emojis, special characters

### 2️⃣ Common Preprocessing Steps

| Step | Description | Example |
|------|-------------|---------|
| **Lowercasing** | Convert all text to lowercase | "Hello World" → "hello world" |
| **Remove Punctuation** | Remove special characters | "Hello!!" → "Hello" |
| **Remove Stopwords** | Remove common words | "the quick brown fox" → "quick brown fox" |
| **Tokenization** | Split into words/tokens | "I love Python" → ["I", "love", "Python"] |
| **Stemming** | Reduce to root form | "running" → "run" |
| **Lemmatization** | Reduce to dictionary form | "better" → "good" |
| **Handle Numbers** | Remove or transform | "3 apples" → "three apples" or "apples" |

### 3️⃣ Text Encoding Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **Bag-of-Words (BoW)** | Count word occurrences | Simple classification |
| **TF-IDF** | Term frequency-inverse document frequency | Search relevance, classification |
| **Word2Vec** | Word embeddings | Semantic similarity |
| **GloVe** | Global vectors for word representation | Language modeling |
| **BERT Embeddings** | Contextual embeddings | Advanced NLP tasks |
| **Character Encoding** | Encode individual characters | RNNs, sequence models |

### 4️⃣ Python Pipeline Example

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Sample data
data = pd.DataFrame({'text': ["I love Python programming!", "Machine learning is amazing."]})

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
data['clean_text'] = data['text'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_text'])
```

### 5️⃣ Notes for ML/Analytics
- Text must be converted to numeric form before ML
- Classical ML: BoW, TF-IDF
- Deep Learning: Word embeddings, Transformers
- Derived features: text length, sentiment score, readability metrics

---

## 📘 Handling Geospatial & Identifier Data

### 1️⃣ Understanding Geospatial & Identifiers
- **Identifiers**: Unique keys (CustomerID, ProductID, Email)
- **Geospatial**: Location data (coordinates, addresses, regions)

### 2️⃣ Common Preprocessing Steps

| Step | Description | Example |
|------|-------------|---------|
| **Drop Identifiers** | Remove IDs for ML | CustomerID → remove |
| **Aggregate Features** | Create summary stats | Purchases per customer |
| **Coordinate Processing** | Handle lat/long | Normalize, cluster |
| **Distance Calculation** | Compute distances | Haversine formula |
| **Region Encoding** | Map locations to regions | City → State → Country |
| **Spatial Features** | External data integration | Population density by zipcode |

### 3️⃣ Usage in ML vs Visualization

| Data Type | Machine Learning | Data Visualization |
|-----------|------------------|-------------------|
| **Identifiers** | ❌ Drop (not predictive) | ✅ Labels, grouping |
| **Geospatial** | ⚠️ Transform to features | ✅ Maps, heatmaps |

### 4️⃣ Notes for ML/Analytics
- Identifiers: useful for joins, but drop for modeling
- Geospatial: engineer features (distances, clusters, densities)
- Visualization: crucial for maps and spatial analysis

---

## 📘 Handling Multimedia Data (Images, Audio, Video)

### 1️⃣ Understanding Multimedia Data
- **Images**: 2D/3D pixel arrays (RGB, grayscale)
- **Audio**: 1D time-series signals or spectrograms
- **Video**: Sequence of image frames with audio

### 2️⃣ Common Preprocessing Steps

| Data Type | Preprocessing Steps | Examples |
|-----------|---------------------|----------|
| **Images** | Resize, normalize, augment, color space conversion | 224x224 resize, /255 normalization |
| **Audio** | Resample, filter, spectrogram, MFCC features | 16kHz sampling, noise removal |
| **Video** | Frame extraction, optical flow, temporal sampling | 30fps to 5fps sampling |

### 3️⃣ Feature Extraction Methods

| Data Type | Feature Extraction | Models |
|-----------|---------------------|---------|
| **Images** | CNN features, HOG, SIFT | ResNet, VGG, EfficientNet |
| **Audio** | MFCC, Spectrograms, Waveforms | LSTM, CNN, Transformers |
| **Video** | 3D CNN, RNN on frames | I3D, SlowFast, TimeSformer |

### 4️⃣ Python Examples

```python
# Image Preprocessing
from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# Audio Preprocessing
import librosa
import librosa.display

def preprocess_audio(file_path, sr=16000):
    y, sr = librosa.load(file_path, sr=sr)
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc
```

### 5️⃣ Notes for ML/Analytics
- **Storage Intensive**: Requires significant memory/disk space
- **Computationally Expensive**: Needs GPU acceleration
- **Data Augmentation**: Crucial for preventing overfitting
- **Transfer Learning**: Use pretrained models for better performance

---

## 🎯 Summary Table: Data Type Preprocessing Guide

| Data Type | ML Preprocessing | Key Methods | Common Models |
|-----------|------------------|-------------|---------------|
| **Numeric** | Scaling, outlier handling | Standardization, Normalization | All models |
| **Categorical** | Encoding | One-Hot, Label, Target Encoding | Tree-based, Linear |
| **Boolean** | Binary conversion | 0/1 mapping | All models |
| **Text** | NLP pipeline | TF-IDF, Embeddings | Naive Bayes, RNNs, Transformers |
| **Geospatial** | Feature engineering | Distance, Clustering | Spatial models, Clustering |
| **Multimedia** | Feature extraction | CNN features, Spectrograms | CNNs, RNNs, Transformers |
| **Structured** | Flattening, parsing | JSON parsing, Time extraction | Domain-specific |

---

## 📊 Quick Reference: When to Use What

### For Traditional ML:
- Numeric: Scale appropriately
- Categorical: One-hot/Label encode
- Text: TF-IDF + Classical models
- Use derived features for performance boost

### For Deep Learning:
- Text: Word embeddings/Transformers
- Images: CNN architectures
- Audio: Spectrogram + CNN/RNN
- Video: 3D CNN/Transformer models

### For Analytics/Visualization:
- Keep identifiers for labeling
- Use geospatial for mapping
- Text for word clouds, sentiment analysis
- Multimedia for exploratory analysis

---

This comprehensive guide covers all major data types you'll encounter in ML and analytics projects. Each section provides specific preprocessing techniques, Python examples, and practical considerations for implementation.
