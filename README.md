# AI Plagiarism Checker using GPT-2

A powerful AI content detection system that distinguishes between human-written, AI-generated, and mixed text using GPT-2 perplexity analysis and linguistic features.

## Features

- ✅ Detects human-written text
- 🤖 Identifies AI-generated content
- ⚠️ Flags mixed/suspicious text (edited AI content)
- 📊 Provides detailed metrics and explanations
- 🎨 Professional Streamlit UI

## Detection Method

The system uses a multi-feature approach:

1. **GPT-2 Perplexity**: Measures how "surprised" the language model is by the text
2. **Burstiness**: Analyzes sentence length variance (humans vary more)
3. **Repetition Score**: Detects n-gram patterns
4. **Lexical Diversity**: Measures vocabulary richness
5. **Random Forest Classifier**: Combines all features for final prediction

## Installation

### 1. Clone/Download the Project

```bash
cd ai_plagiarism_checker
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Generate Dataset

Generate training data (human, AI, and mixed texts):

```bash
python dataset_generator.py
```

This creates three files in the `data/` directory:
- `human.txt` - Human-written samples
- `ai.txt` - AI-generated samples  
- `mixed.txt` - Mixed/edited samples

**Note**: AI text generation takes 5-10 minutes.

### Step 2: Train the Classifier

Train the Random Forest model:

```bash
python train_classifier.py
```

This will:
- Extract features from all samples
- Train the classifier
- Display evaluation metrics
- Save the model as `model.pkl`

Expected accuracy: 85-95%

### Step 3: Run the Application

Launch the Streamlit UI:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Project Structure

```
ai_plagiarism_checker/
│
├── app.py                  # Streamlit UI
├── detector.py             # GPT-2 perplexity logic
├── features.py             # NLP feature extraction
├── train_classifier.py     # Model training
├── dataset_generator.py    # Dataset creation
├── model.pkl               # Trained classifier (generated)
├── requirements.txt        # Dependencies
├── README.md              # This file
│
└── data/                  # Generated datasets
    ├── human.txt
    ├── ai.txt
    └── mixed.txt
```

## How It Works

1. **Input**: User provides text through the UI
2. **Perplexity Calculation**: GPT-2 evaluates the text
3. **Feature Extraction**: Multiple linguistic metrics are computed
4. **Classification**: Random Forest predicts the category
5. **Output**: Results with confidence scores and detailed metrics

## Metrics Explanation

- **Perplexity**: Lower values suggest AI-generated text (AI is more predictable)
- **Burstiness**: Higher values indicate human writing (more varied sentence lengths)
- **Repetition**: Detects repetitive patterns common in AI text
- **Lexical Diversity**: Type-Token Ratio measuring vocabulary richness

## Technical Details

- **Model**: GPT-2 (124M parameters)
- **Classifier**: Random Forest (100 estimators)
- **Features**: 6 linguistic metrics
- **Training Size**: ~300 samples (100 per class)
- **Framework**: PyTorch, Transformers, Scikit-learn

## Use Cases

- Academic integrity verification
- Content authenticity checking
- Detecting AI-edited essays
- Research on AI-generated text

## Limitations

- Requires minimum 50 characters for accurate detection
- Works best with English text
- May misclassify heavily edited human text
- Accuracy depends on training data quality

## Future Improvements

- Multi-language support
- Fine-tuned detection models
- Larger training dataset
- Real-time batch processing
- API endpoint for integration

## Credits

**B.Tech CSE Final Year Project**

Technologies:
- GPT-2 (HuggingFace)
- Scikit-learn
- NLTK
- Streamlit
- PyTorch

## License

Educational/Academic Use

---

For questions or issues, please refer to the documentation or contact the development team.
