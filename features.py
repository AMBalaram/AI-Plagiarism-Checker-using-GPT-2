"""
NLP Feature Extraction
Extracts linguistic features for AI detection
"""

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import re

def _ensure_nltk_tokenizers():
    """Ensure required NLTK tokenizers are available (punkt and punkt_tab)."""
    # Try punkt_tab first (some NLTK versions use this)
    try:
        nltk.data.find('tokenizers/punkt_tab/english')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass

    # Fallback to standard punkt
    try:
        nltk.data.find('tokenizers/punkt/english.pickle')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass

# Run the check once at import time
_ensure_nltk_tokenizers()

class FeatureExtractor:
    def __init__(self):
        pass
    
    def calculate_burstiness(self, text):
        """
        Calculate burstiness (sentence length variance)
        AI text often has more uniform sentence lengths
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) < 2:
            return 0.0
        
        lengths = [len(word_tokenize(s)) for s in sentences]
        
        if len(lengths) == 0:
            return 0.0
        
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        # Coefficient of variation
        if mean_length == 0:
            return 0.0
        
        burstiness = std_length / mean_length
        return burstiness
    
    def calculate_repetition_score(self, text):
        """
        Calculate repetition based on n-grams
        AI text may have more repetitive patterns
        """
        words = word_tokenize(text.lower())
        
        if len(words) < 4:
            return 0.0
        
        # Bigrams and trigrams
        bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        
        # Count repetitions
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)
        
        # Repetition ratio
        total_bigrams = len(bigrams)
        total_trigrams = len(trigrams)
        
        repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)
        repeated_trigrams = sum(1 for count in trigram_counts.values() if count > 1)
        
        repetition_score = 0
        if total_bigrams > 0:
            repetition_score += repeated_bigrams / total_bigrams
        if total_trigrams > 0:
            repetition_score += repeated_trigrams / total_trigrams
        
        return repetition_score / 2
    
    def calculate_lexical_diversity(self, text):
        """
        Calculate lexical diversity (Type-Token Ratio)
        Ratio of unique words to total words
        """
        words = word_tokenize(text.lower())
        
        # Remove punctuation
        words = [w for w in words if w.isalnum()]
        
        if len(words) == 0:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        # Type-Token Ratio
        ttr = unique_words / total_words
        
        return ttr
    
    def extract_all_features(self, text, perplexity):
        """
        Extract all features for a given text
        Returns feature vector
        """
        features = {
            'perplexity': perplexity,
            'burstiness': self.calculate_burstiness(text),
            'repetition': self.calculate_repetition_score(text),
            'lexical_diversity': self.calculate_lexical_diversity(text),
            'avg_sentence_length': self._avg_sentence_length(text),
            'word_count': len(word_tokenize(text))
        }
        
        return features
    
    def _avg_sentence_length(self, text):
        """Calculate average sentence length in words"""
        sentences = sent_tokenize(text)
        
        if len(sentences) == 0:
            return 0.0
        
        lengths = [len(word_tokenize(s)) for s in sentences]
        return np.mean(lengths)
    
    def features_to_vector(self, features_dict):
        """Convert feature dictionary to numpy array"""
        return np.array([
            features_dict['perplexity'],
            features_dict['burstiness'],
            features_dict['repetition'],
            features_dict['lexical_diversity'],
            features_dict['avg_sentence_length'],
            features_dict['word_count']
        ]).reshape(1, -1)
