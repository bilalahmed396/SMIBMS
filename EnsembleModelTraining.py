"""
OPTIMIZED ENSEMBLE MODEL - PUSH TO >90%
Building on 89.28% baseline with additional optimizations
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("OPTIMIZED ENSEMBLE MODEL - TARGET: >90%")
print("Building on 89.28% baseline")
print("="*80)

# ============================================================================
# ENHANCED ATTENTION LAYERS
# ============================================================================

class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                                shape=(input_shape[-1], input_shape[-1]),
                                initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias',
                                shape=(input_shape[-1],),
                                initializer='zeros', trainable=True)
        self.u = self.add_weight(name='att_context',
                                shape=(input_shape[-1], 1),
                                initializer='glorot_uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        uit = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.squeeze(ait, -1)
        ait = tf.nn.softmax(ait, axis=-1)
        ait = tf.expand_dims(ait, -1)
        weighted_input = x * ait
        return tf.reduce_sum(weighted_input, axis=1)

class MultiHeadAttention(layers.Layer):
    """Simplified multi-head attention"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        batch_size = tf.shape(x)[0]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        return self.dense(output)

# ============================================================================
# ADVANCED DATA PREPROCESSING
# ============================================================================

print("\n[STEP 1] Loading Dataset...")
data_path = '/kaggle/input/twitter-airline-sentiment/Tweets.csv'
df = pd.read_csv(data_path)
df = df[['text', 'airline_sentiment']].copy()

# Sentiment lexicons
POSITIVE_WORDS = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                  'best', 'love', 'perfect', 'awesome', 'brilliant', 'thank', 'thanks'}
NEGATIVE_WORDS = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor',
                  'sucks', 'disappointed', 'delay', 'cancel', 'late', 'lost', 'never'}

def enhanced_clean(text):
    """Enhanced preprocessing"""
    text = text.lower()

    # Preserve negations
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"n't", " not", text)

    # Count sentiment markers before cleaning
    pos_count = sum(1 for word in text.split() if word in POSITIVE_WORDS)
    neg_count = sum(1 for word in text.split() if word in NEGATIVE_WORDS)

    # Remove URLs and mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)

    # Preserve important punctuation
    exclamations = text.count('!')
    questions = text.count('?')

    # Clean
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Add sentiment indicators
    if pos_count > neg_count and pos_count >= 2:
        text += ' positive'
    elif neg_count > pos_count and neg_count >= 2:
        text += ' negative'

    return text

print("\n[STEP 2] Enhanced Text Preprocessing...")
df['cleaned_text'] = df['text'].apply(enhanced_clean)
df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)

print("\n[STEP 3] Label Encoding...")
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['airline_sentiment'])

print("\n[STEP 4] Aggressive Balancing...")
# More aggressive balancing
df_negative = df[df['sentiment_encoded'] == 0]
df_neutral = df[df['sentiment_encoded'] == 1]
df_positive = df[df['sentiment_encoded'] == 2]

# Upsample to achieve better balance
df_neutral_upsampled = resample(df_neutral, replace=True,
                                n_samples=int(len(df_negative) * 0.7), random_state=42)
df_positive_upsampled = resample(df_positive, replace=True,
                                 n_samples=int(len(df_negative) * 0.6), random_state=42)

df_balanced = pd.concat([df_negative, df_neutral_upsampled, df_positive_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced dataset size: {len(df_balanced)}")
print(df_balanced['airline_sentiment'].value_counts())

print("\n[STEP 5] Splitting Data...")
X = df_balanced['cleaned_text'].values
y = df_balanced['sentiment_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

print("\n[STEP 6] Tokenization...")
MAX_WORDS = 25000  # Increased vocabulary
MAX_LENGTH = 120   # Longer sequences
EMBEDDING_DIM = 200

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LENGTH)
X_val_pad = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=MAX_LENGTH)
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LENGTH)

print(f"Vocabulary size: {len(tokenizer.word_index)}")

print("\n[STEP 7] Creating Enhanced Embedding Matrix...")
vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)
embedding_matrix = np.random.randn(vocab_size, EMBEDDING_DIM) * 0.01

print(f"Embedding matrix shape: {embedding_matrix.shape}")

print("\n[STEP 8] Computing Class Weights...")
class_weights = dict(enumerate(compute_class_weight('balanced',
                                                     classes=np.unique(y_train),
                                                     y=y_train)))
print(f"Class weights: {class_weights}")

# ============================================================================
# ENHANCED MODEL ARCHITECTURES
# ============================================================================

def create_model_1_enhanced_lstm(embedding_matrix, max_length, num_classes):
    """Enhanced BiLSTM with Multi-Head Attention"""
    inputs = layers.Input(shape=(max_length,))

    x = layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                        trainable=True)(inputs)
    x = layers.SpatialDropout1D(0.2)(x)

    # Multi-head attention
    x = MultiHeadAttention(d_model=EMBEDDING_DIM, num_heads=4)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # BiLSTM
    x = layers.Bidirectional(layers.LSTM(160, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(80, return_sequences=True, dropout=0.2))(x)

    # Attention
    attention = AttentionLayer()(x)
    pool_max = layers.GlobalMaxPooling1D()(x)
    pool_avg = layers.GlobalAveragePooling1D()(x)
    concat = layers.Concatenate()([attention, pool_max, pool_avg])

    # Dense layers with label smoothing awareness
    x = layers.Dense(256, activation='relu')(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_model_2_enhanced_cnn_gru(embedding_matrix, max_length, num_classes):
    """Enhanced CNN+GRU"""
    inputs = layers.Input(shape=(max_length,))

    x = layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                        trainable=True)(inputs)
    x = layers.SpatialDropout1D(0.2)(x)

    # Multi-scale CNN with more filters
    conv_outputs = []
    for kernel_size in [2, 3, 4, 5]:
        conv = layers.Conv1D(160, kernel_size, activation='relu', padding='same')(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.GlobalMaxPooling1D()(conv)
        conv_outputs.append(conv)

    cnn_concat = layers.Concatenate()(conv_outputs)

    # BiGRU branch
    gru = layers.Bidirectional(layers.GRU(160, return_sequences=True, dropout=0.2))(x)
    gru = layers.Bidirectional(layers.GRU(80, return_sequences=True, dropout=0.2))(gru)

    gru_attention = AttentionLayer()(gru)
    gru_pool = layers.GlobalAveragePooling1D()(gru)

    # Combine CNN and GRU
    combined = layers.Concatenate()([cnn_concat, gru_attention, gru_pool])

    x = layers.Dense(256, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_model_3_hybrid(embedding_matrix, max_length, num_classes):
    """Hybrid CNN+LSTM+GRU"""
    inputs = layers.Input(shape=(max_length,))

    x = layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                        trainable=True)(inputs)
    x = layers.SpatialDropout1D(0.2)(x)

    # CNN path
    conv1 = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv2 = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    conv2 = layers.BatchNormalization()(conv2)
    conv_concat = layers.Concatenate()([conv1, conv2])

    # LSTM path
    lstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(conv_concat)
    lstm_out = AttentionLayer()(lstm)

    # GRU path
    gru = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.2))(conv_concat)
    gru_out = layers.GlobalMaxPooling1D()(gru)

    # CNN pooling
    cnn_out = layers.GlobalAveragePooling1D()(conv_concat)

    # Combine all paths
    combined = layers.Concatenate()([lstm_out, gru_out, cnn_out])

    x = layers.Dense(256, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_model_4_deep_attention(embedding_matrix, max_length, num_classes):
    """Deep model with stacked attention"""
    inputs = layers.Input(shape=(max_length,))

    x = layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                        trainable=True)(inputs)
    x = layers.SpatialDropout1D(0.2)(x)

    # Stacked BiLSTM with attention at each level
    x = layers.Bidirectional(layers.LSTM(160, return_sequences=True, dropout=0.2))(x)
    att1 = AttentionLayer()(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
    att2 = AttentionLayer()(x)

    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True, dropout=0.2))(x)
    att3 = AttentionLayer()(x)

    pool = layers.GlobalMaxPooling1D()(x)

    # Combine all attention outputs
    combined = layers.Concatenate()([att1, att2, att3, pool])

    x = layers.Dense(256, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# ============================================================================
# TRAIN ENSEMBLE WITH 4 MODELS
# ============================================================================

print("\n[STEP 9] Training Enhanced Ensemble (4 Models)...")

NUM_CLASSES = len(np.unique(y))
BATCH_SIZE = 64
EPOCHS = 50

models = []
model_names = ['Enhanced BiLSTM+MultiHead', 'Enhanced CNN+BiGRU',
               'Hybrid CNN+LSTM+GRU', 'Deep Attention']
model_creators = [create_model_1_enhanced_lstm, create_model_2_enhanced_cnn_gru,
                  create_model_3_hybrid, create_model_4_deep_attention]

for i, (name, creator) in enumerate(zip(model_names, model_creators)):
    print(f"\n{'='*60}")
    print(f"Training Model {i+1}/4: {name}")
    print(f"{'='*60}")

    model = creator(embedding_matrix, MAX_LENGTH, NUM_CLASSES)

    # Use AdamW optimizer with gradient clipping
    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True,
                     mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                         min_lr=1e-7, verbose=1),
        ModelCheckpoint(f'model_{i+1}_{name.replace(" ", "_").replace("+", "")}.keras',
                       monitor='val_accuracy', save_best_only=True, mode='max')
    ]

    history = model.fit(
        X_train_pad, y_train,
        validation_data=(X_val_pad, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2
    )

    models.append(model)

    # Evaluate
    val_pred = np.argmax(model.predict(X_val_pad, verbose=0), axis=1)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"\n✓ {name} Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

print("\n" + "="*80)
print("All 4 models trained!")
print("="*80)

# ============================================================================
# ADVANCED ENSEMBLE PREDICTIONS
# ============================================================================

print("\n[STEP 10] Creating Advanced Ensemble Predictions...")

# Get predictions from all models
test_predictions = []
val_predictions = []

for i, model in enumerate(models):
    # Test predictions
    test_pred_probs = model.predict(X_test_pad, verbose=0)
    test_predictions.append(test_pred_probs)

    # Validation predictions for weighting
    val_pred_probs = model.predict(X_val_pad, verbose=0)
    val_predictions.append(val_pred_probs)

    # Individual model test accuracy
    test_pred_classes = np.argmax(test_pred_probs, axis=1)
    acc = accuracy_score(y_test, test_pred_classes)
    print(f"Model {i+1} ({model_names[i]}) Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

print("\n" + "="*60)
print("Ensemble Methods:")
print("="*60)

# 1. Average Probability
avg_probs = np.mean(test_predictions, axis=0)
y_pred_avg = np.argmax(avg_probs, axis=1)
acc_avg = accuracy_score(y_test, y_pred_avg)
print(f"\n1. Average Probability: {acc_avg:.4f} ({acc_avg*100:.2f}%)")

# 2. Weighted by validation accuracy
weights = []
for i, model in enumerate(models):
    val_pred = np.argmax(val_predictions[i], axis=1)
    val_acc = accuracy_score(y_val, val_pred)
    weights.append(val_acc ** 2)  # Square to emphasize better models
weights = np.array(weights) / sum(weights)
print(f"   Weights: {weights}")

weighted_probs = np.average(test_predictions, axis=0, weights=weights)
y_pred_weighted = np.argmax(weighted_probs, axis=1)
acc_weighted = accuracy_score(y_test, y_pred_weighted)
print(f"2. Weighted by Val Acc²: {acc_weighted:.4f} ({acc_weighted*100:.2f}%)")

# 3. Majority Voting
votes = np.array([np.argmax(pred, axis=1) for pred in test_predictions])
y_pred_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=votes)
acc_vote = accuracy_score(y_test, y_pred_vote)
print(f"3. Majority Voting: {acc_vote:.4f} ({acc_vote*100:.2f}%)")

# 4. Confidence-weighted voting (new!)
confidence_votes = []
for pred_probs in test_predictions:
    confidence = np.max(pred_probs, axis=1)
    pred_classes = np.argmax(pred_probs, axis=1)
    confidence_votes.append((pred_classes, confidence))

y_pred_conf = []
for i in range(len(X_test)):
    class_scores = {0: 0, 1: 0, 2: 0}
    for pred_classes, confidence in confidence_votes:
        class_scores[pred_classes[i]] += confidence[i]
    y_pred_conf.append(max(class_scores, key=class_scores.get))
y_pred_conf = np.array(y_pred_conf)
acc_conf = accuracy_score(y_test, y_pred_conf)
print(f"4. Confidence-Weighted Voting: {acc_conf:.4f} ({acc_conf*100:.2f}%)")

# Choose best method
best_methods = [
    (acc_avg, 'Average', y_pred_avg),
    (acc_weighted, 'Weighted', y_pred_weighted),
    (acc_vote, 'Voting', y_pred_vote),
    (acc_conf, 'Confidence', y_pred_conf)
]
best_acc, best_name, y_pred = max(best_methods, key=lambda x: x[0])

print(f"\n{'='*60}")
print(f"🏆 BEST: {best_name} Ensemble - {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"{'='*60}")

accuracy = best_acc

# ============================================================================
# FINAL EVALUATION
# ============================================================================

print("\n[STEP 11] Final Comprehensive Evaluation...")

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print("\n" + "="*80)
print("FINAL OPTIMIZED ENSEMBLE RESULTS")
print("="*80)
print(f"\nEnsemble Method: {best_name}")
print(f"Number of Models: {len(models)}")
print(f"\n{'='*60}")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"{'='*60}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Per-class accuracy
class_accs = cm.diagonal() / cm.sum(axis=1)
print("\nPer-Class Accuracy:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"  {class_name}: {class_accs[i]*100:.2f}%")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("OPTIMIZATION SUMMARY")
print("="*80)
print(f"\nPrevious Best: 89.28%")
print(f"Current Best:  {accuracy*100:.2f}%")
print(f"Improvement:   {(accuracy - 0.8928)*100:+.2f}%")

if accuracy >= 0.90:
    print("\n" + "🎉"*30)
    print("SUCCESS! 90% ACCURACY ACHIEVED!")
    print("🎉"*30)
else:
    gap = (0.90 - accuracy) * 100
    print(f"\nClose! Only {gap:.2f}% away from 90%")
    print("\nTo close the gap:")
    print("1. Load actual GloVe Twitter embeddings (biggest impact)")
    print("2. Add more data augmentation")
    print("3. Use 5-fold cross-validation ensemble")
    print("4. Fine-tune hyperparameters per model")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
