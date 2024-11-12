# Movie Review Classification
This project aims to classify movie reviews as positive or negative using various natural language processing (NLP) techniques and machine learning classifiers. The focus is on thorough preprocessing of text data, vectorization, and the application of classifiers with hyperparameter tuning.

## Project Overview
The project leverages various NLP preprocessing techniques to clean and transform movie review text data into structured data suitable for machine learning models. The primary objective is to accurately classify the sentiment of reviews (positive/negative) using optimized classifiers.

## Preprocessing Steps

### Tokenization
Tokenization involves splitting the text into individual words (tokens). This step was performed using the NLTK tokenizer to ensure accurate separation of words, including handling punctuation and special characters.

### Stemming
Two types of stemming were used in this project:

- Custom Stemmer: A tailored stemming function was created to handle various word forms and irregularities in the text. This approach provides flexibility and better customization for specific words and phrases.
- Porter Stemmer: Although evaluated, the custom stemmer was found to perform better and was preferred for further processing.

```python
# Example usage of custom stemmer
gstem = [myStemmerDic(tok, tag, dico) for tok, tag in nltk.pos_tag(sent)]
stems_ours = [stem for tok, stem, tag in gstem]
```

## Lemmatization
Lemmatization was performed using both NLTK and SpaCy, with SpaCy providing more accurate results. Lemmatization ensures that words are reduced to their base or dictionary form, which is important for capturing the true meaning and structure of the text.

## Named Entity Recognition (NER) Replacement
NER was used to replace named entities in the text using SpaCy. This step ensures that names, locations, and other identifiable entities are handled appropriately, which reduces noise and variability in the data.

## TF-IDF Vectorization
The TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer was employed to transform text data into numerical representations. Key configurations include:

- 1-grams, 2-grams, etc.: Different n-grams were explored to capture context and phrase structures.
- Stopwords and Punctuation Removal: Common stopwords and punctuation were removed to improve signal-to-noise ratio.
- min_df Parameter: Used to filter out terms that appear in fewer documents, thus reducing the vector size.
## Dimensionality Reduction
Dimensionality reduction was performed using Latent Semantic Analysis (LSA) via Truncated Singular Value Decomposition (SVD) to reduce the TF-IDF matrix size while preserving important features.

## Modeling

### Classifiers
The following classifiers were used and evaluated:

- Random Forest Classifier
- Support Vector Machine (SVM)
- Multilayer Perceptron (MLP) Classifier
- Naive Bayes (GaussianNB)

## Hyperparameter Optimization
Hyperparameter tuning was carried out using RandomizedSearchCV and GridSearchCV to find the optimal set of parameters for each classifier.

## Evaluation Metrics
Model performance was assessed using metrics such as:

Accuracy,
Precision,
Recall,
F1 Score,
Cross-validation, including Stratified K-Fold cross-validation, was used to ensure robust model evaluation.

## Usage
Requirements
Python 3.x,
Jupyter Notebook,
NLTK,
SpaCy,
Pandas,
NumPy,
Scikit-learn,
Matplotlib

## Results

The project demonstrates the effectiveness of advanced NLP preprocessing techniques in improving classification accuracy. Comparative results for different models and preprocessing steps are documented within the notebook.
