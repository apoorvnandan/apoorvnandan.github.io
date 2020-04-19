---
layout: post
comments: true
title:  "Training Transformers on TPU"
excerpt: "Training transformers is now super easy thanks to HuggingFace and Tensorflow 2"
date:   2020-04-19 22:00:00
---
This post shows how clean and straightforward it is to use huggingface library and tensorflow 2 to train transformer models on GPU and TPU!

As of today (April, 2020),
this is how you fine tune a BERT model on GPU (Same code works for CPU as well).
```python
# load data
#
# df = load_data()
# -----------------------
# text            , label
# -----------------------
# good job.           1
# oh no!              0
# .....

# tokenize text
text_col = df.text.astype(str)
input_ids = []
for i in range(0, len(text_col), config.chunk_size):
    text_chunk = text_col[i:i+config.chunk_size].tolist()
    encoded = tokenizer.encode_batch(text_chunk)
    input_ids.extend([enc.ids for enc in encoded])

x_train = np.array(input_ids)
y_train = df.label.values

# create data loader
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(config.batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# create model : Input > BERT > Dense
bert = TFBertModel.from_pretrained(config.MODEL_PATH, config=BertConfig())
input_ids = layers.Input(shape=(config.maxlen,), dtype=tf.int32)
sequence_output = bert(input_ids)[0][:, 0, :]
out = layers.Dense(1, activation='sigmoid')(sequence_output)
classifier = models.Model(inputs=input_ids, outputs=out)

# compile
classifier.compile(optimizers.Adam(lr=3e-5), 
                   loss='binary_crossentropy', 
                   metrics=[metrics.AUC()])

# train !
train_history = classifier.fit(
    train_dataset,
    steps_per_epoch=150,
    epochs=30
)
```

And this is how you do the same thing on TPUs. (Few extra lines added to load the model on TPU)
```python
# load data
#
# df = load_data()
# -----------------------
# text            , label
# -----------------------
# good job.           1
# oh no!              0
# .....

# create distribution strategy
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)

# create model : Input > BERT > Dense
with strategy.scope():
    bert = TFBertModel.from_pretrained(config.MODEL_PATH, config=BertConfig())
    input_ids = layers.Input(shape=(config.maxlen,), dtype=tf.int32)
    sequence_output = bert(input_ids)[0][:, 0, :]
    out = layers.Dense(1, activation='sigmoid')(sequence_output)
    classifier = models.Model(inputs=input_ids, outputs=out)
    
    classifier.compile(optimizers.Adam(lr=3e-5), 
                   loss='binary_crossentropy', 
                   metrics=[metrics.AUC()])
    
# tokenize text
text_col = df.text.astype(str)
input_ids = []
for i in tqdm(range(0, len(text_col), config.chunk_size)):
    text_chunk = text_col[i:i+config.chunk_size].tolist()
    encoded = tokenizer.encode_batch(text_chunk)
    input_ids.extend([enc.ids for enc in encoded])

x_train = np.array(input_ids)
y_train = df.label.values

# create data loader
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(config.batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# train !
train_history = classifier.fit(
    train_dataset,
    steps_per_epoch=150,
    epochs=30
)
```

Its honestly amazing how easy this is compared to the old tensorflow 1.1x flow in the original code base of BERT paper!

This abstraction and creation of clean APIs for latest deep learning techniques and models will continue to lower the barrier for using them in applications.
