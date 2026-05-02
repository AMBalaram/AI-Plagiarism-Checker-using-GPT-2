"""
Train Random Forest Classifier
Trains model on human/AI/mixed dataset
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from detector import GPT2Detector
from features import FeatureExtractor
import os

def load_dataset():
    """Load and prepare dataset"""
    print("Loading dataset...")
    
    data = []
    labels = []
    
    # Load human texts (label 0)
    if os.path.exists('data/human.txt'):
        with open('data/human.txt', 'r', encoding='utf-8') as f:
            texts = f.read().split('\n\n')
            texts = [t.strip() for t in texts if len(t.strip()) > 50]
            data.extend(texts)
            labels.extend([0] * len(texts))
        print(f"Loaded {len(texts)} human samples")
    
    # Load AI texts (label 1)
    if os.path.exists('data/ai.txt'):
        with open('data/ai.txt', 'r', encoding='utf-8') as f:
            texts = f.read().split('\n\n')
            texts = [t.strip() for t in texts if len(t.strip()) > 50]
            data.extend(texts)
            labels.extend([1] * len(texts))
        print(f"Loaded {len(texts)} AI samples")
    
    # Load mixed texts (label 2)
    if os.path.exists('data/mixed.txt'):
        with open('data/mixed.txt', 'r', encoding='utf-8') as f:
            texts = f.read().split('\n\n')
            texts = [t.strip() for t in texts if len(t.strip()) > 50]
            data.extend(texts)
            labels.extend([2] * len(texts))
        print(f"Loaded {len(texts)} mixed samples")
    
    return data, labels

def extract_features_from_texts(texts, detector, feature_extractor):
    """Extract features from all texts"""
    print("\nExtracting features from texts...")
    
    feature_vectors = []
    
    for i, text in enumerate(texts):
        # Calculate perplexity
        perplexity = detector.calculate_perplexity(text)
        
        # Extract features
        features = feature_extractor.extract_all_features(text, perplexity)
        feature_vector = feature_extractor.features_to_vector(features)
        feature_vectors.append(feature_vector.flatten())
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(texts)} samples")
    
    return np.array(feature_vectors)

def train_model():
    """Main training function"""
    # Load dataset
    texts, labels = load_dataset()
    
    if len(texts) == 0:
        print("ERROR: No data found. Please run dataset_generator.py first!")
        return
    
    print(f"\nTotal samples: {len(texts)}")
    print(f"Class distribution: {dict(pd.Series(labels).value_counts())}")
    
    # Initialize detector and feature extractor
    detector = GPT2Detector()
    feature_extractor = FeatureExtractor()
    
    # Extract features
    X = extract_features_from_texts(texts, detector, feature_extractor)
    y = np.array(labels)
    
    print(f"\nFeature matrix shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train Random Forest
    print("\nTraining Random Forest Classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    print(f"\nTraining Accuracy: {train_score:.4f}")
    print(f"Testing Accuracy: {test_score:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Detailed metrics
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI', 'Mixed']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_names = ['Perplexity', 'Burstiness', 'Repetition', 'Lexical Diversity', 'Avg Sentence Length', 'Word Count']
    importances = clf.feature_importances_
    
    print("\nFeature Importance:")
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {importance:.4f}")
    
    # Save model
    print("\nSaving model...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    print("\n✓ Model saved as 'model.pkl'")
    print("\nTraining complete! Run 'streamlit run app.py' to use the detector.")

if __name__ == '__main__':
    train_model()
