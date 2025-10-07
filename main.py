
# --- Standard libs & third-party imports ------------------------------------
import os, sys, random, numpy as np, pandas as pd, tensorflow as tf, argparse
from tensorflow import keras as tfk # local alias that wraps legacy tf-keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import TFAutoModel, AutoTokenizer, create_optimizer

# --- Reproducibility & Keras unification ------------------------------------
SEED = 42
os.environ["TF_DETERMINISTIC_OPS"] = "1"        # deterministic GPU ops
os.environ["TF_USE_LEGACY_KERAS"]   = "1"       # make tf.keras the default
sys.modules["keras"] = tfk                      # ensure every ‘import keras’ hits tf-keras
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# 1 ── Load data & cluster-based split
# ---------------------------------------------------------------------------
def cluster_based_split(
    X, y, *, test_size=0.2, val_size=0.2, random_state=None
):
    """
    Cluster-aware train/val/test split for multi-label data.

    Steps
    -----
    1. Run K-Means on the label matrix `y` to capture co-occurrence patterns.
    2. Split clusters into train+val vs. test so those patterns are preserved
       across splits (important for small, imbalanced datasets).
    3. Within the train+val bucket, perform another cluster-wise split to carve
       out a validation set that mirrors train/test distribution.
    Returns
    -------
    Tuple of six NumPy arrays:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    # ---- Cluster labels ----------------------------------------------------
    kmeans   = KMeans(n_clusters=5, random_state=random_state)
    clusters = kmeans.fit_predict(y)
    uniq     = np.unique(clusters)

    # ---- First split: test set --------------------------------------------
    train_val_idx, test_idx = [], []
    for c in uniq:
        idx_c = np.where(clusters == c)[0]          # samples in cluster c
        split = int(len(idx_c) * (1 - test_size))   # keep % for train+val
        train_val_idx.extend(idx_c[:split])
        test_idx      .extend(idx_c[ split:])

    # ---- Second split: validation set -------------------------------------
    clusters_tv = clusters[train_val_idx]
    train_idx, val_idx = [], []
    for c in uniq:
        idx_c = np.where(clusters_tv == c)[0]
        split = int(len(idx_c) * (1 - val_size / (1 - test_size)))
        train_idx.extend(idx_c[:split])
        val_idx  .extend(idx_c[ split:])

    # ---- Slice and return --------------------------------------------------
    X_tv, y_tv = X[train_val_idx], y[train_val_idx]
    return (
        X_tv[train_idx], X_tv[val_idx], X[test_idx],
        y_tv[train_idx], y_tv[val_idx], y[test_idx]
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loading and preprocessing the raw dataset
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("track-a.csv").dropna(subset=['text'])   # keep only rows with text

# The five binary target columns for multi-label emotion classification
emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise']

# X_text → raw sentences; Y → multi-hot label matrix 
X_text = df['text'].to_numpy()
Y      = df[emotion_labels].to_numpy(dtype='float32')

# ─────────────────────────────────────────────────────────────────────────────
# 2. Shuffle once for reproducibility (important before any split)
# ─────────────────────────────────────────────────────────────────────────────
perm = np.random.permutation(len(X_text))
X_text, Y = X_text[perm], Y[perm]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Cluster-aware split: train / validation / test
#    (preserves label-correlation patterns across the splits)
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_val, X_test, y_train, y_val, y_test = cluster_based_split(
    X_text, Y, test_size=0.20, val_size=0.20, random_state=SEED
)

# Many tokenisers expect a list of strings rather than a NumPy array
X_train_text, X_val_text, X_test_text = map(list, [X_train, X_val, X_test])


# ─────────────────────────────────────────────────────────────────────────
# 2-A. Classical baseline: TF-IDF features + one-vs-rest Logistic Regression
# ─────────────────────────────────────────────────────────────────────────

# Vectoriser: bag-of-words with unigrams + bigrams, limited vocab (20k) and
# English stop-word removal. This yields a sparse (n_samples × 20k) matrix.
tfidf = TfidfVectorizer(
    max_features=20_000,
    ngram_range=(1, 2),
    stop_words='english'
)

# Fit on the training split only, then apply the learned vocabulary to val/test.
X_tr_tfidf = tfidf.fit_transform(X_train_text)
X_va_tfidf = tfidf.transform(X_val_text)
X_te_tfidf = tfidf.transform(X_test_text)

# One-vs-rest Logistic Regression:
# - `max_iter=400`   ensures convergence with high-dimensional input
# - `class_weight='balanced'` tackles class imbalance automatically
# - `n_jobs=-1`      leverages all available CPU cores
logreg = OneVsRestClassifier(
    LogisticRegression(
        max_iter=400,
        class_weight='balanced',
        n_jobs=-1
    )
)

# Train the classifier on TF-IDF features
logreg.fit(X_tr_tfidf, y_train)

# Probabilities for later ensembling
val_prob  = {}
test_prob = {}
val_prob['tfidf']  = logreg.predict_proba(X_va_tfidf)
test_prob['tfidf'] = logreg.predict_proba(X_te_tfidf)


# ─────────────────────────────────────────────────────────────────────────
# 2-B. Deep baseline: Bi-LSTM with self-attention
# ─────────────────────────────────────────────────────────────────────────

# -------------------------------
# Text → integer sequence pipeline
# -------------------------------
MAX_NUM_WORDS = 10_000        # keep the 10k most-frequent tokens
MAX_SEQ_LEN   = 100           # clamp / pad every sentence to 100 tokens

tok = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tok.fit_on_texts(X_train_text)                # build the word-index on train only

def to_seq(txt_iterable):
    """Tokenise & left-pad a list of strings → (n_samples × MAX_SEQ_LEN)."""
    return pad_sequences(
        tok.texts_to_sequences(txt_iterable),
        maxlen=MAX_SEQ_LEN
    )

X_tr_seq, X_va_seq, X_te_seq = map(
    to_seq,
    [X_train_text, X_val_text, X_test_text]
)

# -------------------------------
# Class-imbalance-aware loss
# -------------------------------
# Compute per-label positive/negative counts, then derive a weight that
# down-weights the majority class and up-weights the minority class.
pos = tf.reduce_sum(y_train, axis=0)
neg = tf.cast(tf.shape(y_train)[0], tf.float32) - pos
pos_w_tf = neg / (pos + 1e-6) # vector of weights

def weighted_bce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_true, 
        logits=y_pred  # Change to y_pred if using sigmoid
    )  # Shape [batch_size, 5]
    
    # Calculate weights for each sample and class
    weights = y_true * pos_w_tf + (1 - y_true)
    # Apply weights element-wise and take mean
    weighted_bce = tf.reduce_mean(bce * weights)
    return weighted_bce

# -------------------------------
# Model architecture
# -------------------------------
inp = layers.Input(shape=(MAX_SEQ_LEN,))                  # token IDs
x   = layers.Embedding(MAX_NUM_WORDS, 128)(inp)           # 128-d embeddings
x   = layers.Bidirectional(layers.LSTM(
          64, return_sequences=True))(x)                  # Bi-LSTM context
x   = layers.Attention()([x, x])                          # self-attention
x   = layers.GlobalAveragePooling1D()(x)                  # sentence embedding
x   = layers.Dropout(0.5)(x)                              # regularisation
out = layers.Dense(len(emotion_labels), activation='sigmoid')(x)

bilstm = tfk.Model(inp, out)
bilstm.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(1e-3),
    loss=weighted_bce,
    metrics=['accuracy'] 
)

# -------------------------------
# Training
# -------------------------------
bilstm.fit(
    X_tr_seq, y_train,
    validation_data=(X_va_seq, y_val),
    epochs=15,
    batch_size=64,
    verbose=0,   # silence per-epoch logs
    callbacks=[tfk.callbacks.EarlyStopping(
        patience=2, restore_best_weights=True)]
)

# ─────────────────────────────────────────────────────────────────────────
# 2-C. Transformer baseline: DistilBERT fine-tuning
# ─────────────────────────────────────────────────────────────────────────
HF_MODEL = "distilbert-base-uncased"

# Hugging Face tokenizer → encodes text to (input_ids, attention_mask)
hf_tok = AutoTokenizer.from_pretrained(HF_MODEL)

def bert_enc(txt_iterable):
    """Tokenise, truncate/pad to 128 WordPieces, return TensorFlow tensors."""
    return hf_tok(
        txt_iterable,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="tf"
    )

# Encode each split once to avoid on-the-fly overhead
tr_enc, va_enc, te_enc = map(bert_enc, [X_train_text, X_val_text, X_test_text])

# ---- Model head on top of DistilBERT’s [CLS] token ------------------------
transformer = TFAutoModel.from_pretrained(HF_MODEL)

ids  = tfk.Input(shape=(None,), dtype=tf.int32, name='input_ids')
mask = tfk.Input(shape=(None,), dtype=tf.int32, name='attention_mask')
cls  = transformer(ids, attention_mask=mask)[0][:, 0, :]      # [CLS] vector
hidd = layers.Dense(256, activation='relu')(cls)
hidd = layers.Dropout(0.3)(hidd)
logits = layers.Dense(len(emotion_labels), activation='sigmoid')(hidd)

bert = tfk.Model([ids, mask], logits)

# AdamW schedule from HF utility: warm-up ⅓ of total steps
steps = len(X_train_text) // 32 * 3
opt, _ = create_optimizer(2e-5, steps, steps // 3)

bert.compile(optimizer=opt, loss=weighted_bce)

# Fine-tune for three epochs (early-stop not critical here: small #epochs)
bert.fit(
    dict(tr_enc), y_train,
    validation_data=(dict(va_enc), y_val),
    epochs=3,
    batch_size=32,
    verbose=0
)

# Cache probabilities for the ensemble
val_prob['bert']  = bert.predict(dict(va_enc), batch_size=64, verbose=0)
test_prob['bert'] = bert.predict(dict(te_enc), batch_size=64, verbose=0)

# ─────────────────────────────────────────────────────────────────────────
# 3. Soft-vote ensemble + per-label threshold tuning
# ─────────────────────────────────────────────────────────────────────────
# Stack [model × sample × label] → average probabilities across models
val_stack  = np.stack(list(val_prob.values()),  axis=0)
test_stack = np.stack(list(test_prob.values()), axis=0)
val_mean   = val_stack.mean(axis=0)
test_mean  = test_stack.mean(axis=0)

# Grid-search a separate decision threshold for each emotion
thr_grid = np.linspace(0.05, 0.95, 19)
best_thr = np.zeros(len(emotion_labels))

for j, lbl in enumerate(emotion_labels):
    best_f1, best_t = 0.0, 0.5
    for t in thr_grid:
        f1 = f1_score(y_val[:, j], (val_mean[:, j] >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    best_thr[j] = best_t

# ─────────────────────────────────────────────────────────────────────────
# 4. Final evaluation on the hold-out test set
# ─────────────────────────────────────────────────────────────────────────
y_pred = (test_mean >= best_thr).astype(int)

print(classification_report(y_test, y_pred,
                            target_names=emotion_labels, digits=3))
print("Macro-F1 :", f1_score(y_test, y_pred, average='macro'))

# ─────────────────────────────────────────────────────────────────────────
# 5. Inference helpers  (single text or list of texts)
# ─────────────────────────────────────────────────────────────────────────
def _prep_seq(texts):
    """Helper → Bi-LSTM: text → padded integer sequence."""
    return pad_sequences(tok.texts_to_sequences(texts), maxlen=100)

def _prep_bert(texts):
    """Helper → DistilBERT: text → dict(input_ids, attention_mask)."""
    return dict(hf_tok(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="tf"
    ))

def predict(texts):
    """
    Ensemble inference.

    Parameters
    ----------
    texts : str | list[str]
        One string or a list of strings to classify.

    Returns
    -------
    p_mean : np.ndarray
        Averaged probability matrix  (n_samples × n_labels).
    label_lists : list[list[str]]
        Human-readable predictions: list of emotion tags per sample.
    """
    if isinstance(texts, str):
        texts = [texts]

    # --- Individual model probabilities ------------------------------------
    p_lr   = logreg.predict_proba(tfidf.transform(texts))
    p_lstm = bilstm.predict(_prep_seq(texts), batch_size=256)
    p_bert = bert.predict(_prep_bert(texts), batch_size=64)

    # --- Soft-vote ----------------------------------------------------------
    p_mean = np.stack([p_lr, p_lstm, p_bert]).mean(axis=0)
    y_hat  = (p_mean >= best_thr).astype(bool)

    # Map boolean mask → list of emotion strings
    label_lists = [
        [emotion_labels[j] for j, flag in enumerate(row) if flag]
        for row in y_hat
    ]
    return p_mean, label_lists



# ─────────────────────────────────────────────────────────────────────────
# CLI entry point:   python this_script.py path/to/your.csv
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch predict emotions from CSV")
    ap.add_argument("csv", help="Input CSV path")
    args = ap.parse_args()

    csv_path = args.csv
    df = pd.read_csv(csv_path)

    if 'text' not in df.columns:
        raise ValueError("Expected the CSV to contain a 'text' column with text data.")

    # Batch prediction using the same helper
    _, label_lists = predict(df['text'].tolist())
    df['predicted_labels'] = label_lists

    out_path = os.path.splitext(csv_path)[0] + "_predicted.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved predictions to {out_path}")
    df.head()        # prints DataFrame head when run interactively
