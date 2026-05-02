"""
GPT-2 Perplexity Detector
Calculates perplexity of text using GPT-2 language model
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

class GPT2Detector:
    def __init__(self):
        """Initialize GPT-2 model for perplexity calculation"""
        print("Loading GPT-2 model...")
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model.eval()
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"GPT-2 loaded on {self.device}")
    
    def calculate_perplexity(self, text):
        """
        Calculate perplexity of text using GPT-2
        Lower perplexity often indicates AI-generated text
        """
        if not text or len(text.strip()) < 10:
            return float('inf')
        
        # Tokenize
        encodings = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
        input_ids = encodings.input_ids.to(self.device)
        
        # Calculate loss
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def calculate_perplexity_per_sentence(self, text):
        """Calculate perplexity for each sentence"""
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        
        if not sentences:
            return []
        
        perplexities = []
        for sentence in sentences:
            try:
                ppl = self.calculate_perplexity(sentence + '.')
                perplexities.append(ppl)
            except:
                continue
        
        return perplexities
