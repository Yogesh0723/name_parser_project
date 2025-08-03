import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

class NameDataGenerator(Sequence):
    def __init__(self, sentences, labels, word2idx, tag2idx, max_len=30, batch_size=128, shuffle=True):
        self.sentences = sentences
        self.labels = labels
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(sentences))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.sentences) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = []
        y = []
        for i in batch_indices:
            tokens = self.sentences[i]
            tags = self.labels[i]
            seq = [self.word2idx.get(w, self.word2idx["UNK"]) for w in tokens]
            tag_seq = [self.tag2idx[t] for t in tags]
            X.append(seq)
            y.append(tag_seq)

        X = pad_sequences(X, maxlen=self.max_len, padding="post", value=self.word2idx["PAD"])
        y = pad_sequences(y, maxlen=self.max_len, padding="post", value=self.tag2idx["B-SUR"])  # Default fallback

        y = np.array([to_categorical(seq, num_classes=len(self.tag2idx)) for seq in y])
        return np.array(X), y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
