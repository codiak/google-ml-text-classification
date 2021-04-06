import load_data, explore_data, vectorize_data
from train_ngram_model import train_ngram_model


# Step 1: Gather Data
# -------------------

"""
Create a /data folder in this repo
Download v1 dataset to /data folder from
https://ai.stanford.edu/~amaas/data/sentiment/
Extract contents into /data/aclImdb
"""

# Step 2: Explore Data
# --------------------

# Load the dataset
data_dir = './data/'
data_tuple = load_data.load_imdb_sentiment_analysis_dataset(data_dir)
(train_texts, train_labels), (val_texts, val_labels) = data_tuple
# explore_data.get_num_words_per_sample(train_texts)

# The two charts in the course
# explore_data.plot_frequency_distribution_of_ngrams(train_texts)
# explore_data.plot_sample_length_distribution(train_texts)

# explore_data.plot_class_distribution(train_labels)

# Train
# acc, loss = train_ngram_model.train_ngram_model(data)

# Step 3: Prepare Data
# --------------------

# N-gram Tokenization into unigrams and bigrams
# Vectorize using tf-idf encoding
# Apply feature selection (top 20,000 features)
data_vectors = vectorize_data.ngram_vectorize(train_texts, train_labels, val_texts)

# Step 4: Build, Train, and Evaluate
# ----------------------------------

LEARNING_RATE = 1e-3
EPOCHS = 1000
BATCH_SIZE = 128
LAYERS = 2
UNITS = 64
DROPOUT_RATE = 0.2
model_eval = train_ngram_model(data_tuple, LEARNING_RATE, EPOCHS,
                              BATCH_SIZE, LAYERS, UNITS, DROPOUT_RATE)
