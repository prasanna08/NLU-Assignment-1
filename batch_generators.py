import numpy as np
from collections import Counter, defaultdict
import json

def preprocess_corpus(corpus_words, ignore_threshold=5):
    # Consider only alphabetic words.
    words = [wrd.lower() for wrd in corpus_words if wrd.isalpha()]
    word_freq = Counter(words)
    
    # Ignore words which occur less than threshold times in all documents.
    # Replace them by a special UNK word.
    if ignore_threshold > 0:
        unk_word = 'UNK'
        unk_cnt = 0
        unique_words = list(word_freq.keys())
        for wrd in unique_words:
            if word_freq[wrd] < ignore_threshold:
                unk_cnt += word_freq[wrd]
                del word_freq[wrd]
    
    # Create a Word to ID map.
    total_words = sum(cnt for wrd, cnt in word_freq.items())
    increment = 1 if ignore_threshold > 0 else 0
    word_to_id = {wrd: (i+increment) for i, wrd in enumerate(word_freq)}
    if ignore_threshold > 0:
        word_to_id[unk_word] = 0
    
    # Generate Pn and Ps probabilities.
    total_pn = sum(np.power(float(word_freq[wrd])/total_words, 0.75) for wrd in word_freq)
    word_metadata = {
        word_to_id[wrd]: {
            'freq': word_freq[wrd],
            'Uw': float(word_freq[wrd])/total_words,
            'Ps': 1 - np.sqrt(1e-5 / (float(word_freq[wrd])/total_words)),
            'Pn': np.power(float(word_freq[wrd])/total_words, 0.75) / total_pn,
        } for wrd in word_freq
    }
    
    if ignore_threshold > 0:
        word_metadata[0] = {
            'freq': unk_cnt, 'Uw': 0, 'Ps': 1 - np.sqrt(1e-5 / unk_cnt) , 'Pn': 0
        }
    
    # For numerical stability, due to insufficient precision in float.
    cdf = 0.0
    for wrd in word_metadata:
        cdf += word_metadata[wrd]['Pn']
        word_metadata[wrd]['cdf'] = cdf
    return word_to_id, word_metadata

class BatchGenerator(object):
    def __init__(self, corpus, batch_size, split_ratio=0.8, window_size=3, loaded_data=None):
        self.corpus = corpus
        self.ids = corpus.fileids()
        self.batch_size = batch_size
        self.corpus_words = [wrd.lower() for wrd in self.corpus.words() if wrd.isalpha()]
        if loaded_data is None:
            self.word_to_id, self.word_metadata = preprocess_corpus(self.corpus_words)
        else:
            self.word_to_id = loaded_data['word_to_id']
            self.word_metadata = {int(k): v for (k, v) in loaded_data['word_metadata'].items()}
        self.n_words = len(self.word_to_id)
        self.split_ratio = split_ratio
        self.window_size = window_size
        self.n_steps = 0
        self.splitted = False
        self._cursor = self.window_size + 1
        self.word_ids = [self.word_to_id[wrd] for wrd in self.corpus_words if wrd in self.word_to_id]
    
    def set_batch_size(self, new_size):
        self.batch_size = new_size
    
    def split(self):
        self.shuffle_data()
        self.splitted = True
        instances = len(self.ids)
        i = int(self.split_ratio*instances)
        self.train_ids = self.ids[:i]
        self.valid_ids = self.ids[i:]
        self.train_word_ids = [self.word_to_id[wrd] for wrd in self.corpus.words(self.train_ids) if wrd in self.word_to_id]
        self.n_train_words = len(self.train_word_ids)
        self.validation_word_ids = [self.word_to_id[wrd] for wrd in self.corpus.words(self.train_ids) if wrd in self.word_to_id]
        self.n_validation_words = len(self.validation_word_ids)
    
    def get_validation_data(self):
        if self.splitted:
            return self.valid_data, self.valid_labels
        else:
            return None
    
    def __iter__(self):
        self._cursor = self.window_size + 1
        return self
    
    def get_negative_probs(self):
        return [self.word_metadata[wrd]['Pn'] for wrd in sorted(self.word_metadata.keys())]
    
    def _increment_cursor(self, max_len):
        self._cursor = (self._cursor + 1) % (max_len - self.window_size)
        if self._cursor == 0:
            self._cursor = self.window_size + 1
            return True
        return False
    
    @classmethod
    def load_from_metadata(cls, fname, corpus, batch_size, split_ratio=0.8, window_size=3):
        f = open(fname, 'r')
        data = json.loads(f.read())
        f.close()
        return cls(corpus, batch_size, split_ratio=split_ratio, window_size=window_size, loaded_data=data)
        
    def store_metadata(self, fname):
        data = {
            'word_to_id': self.word_to_id,
            'word_metadata': self.word_metadata
        }
        f = open(fname, 'w')
        f.write(json.dumps(data))
        f.close()

    def __next__(self):
        word_ids = self.word_ids if not self.splitted else self.train_word_ids
        batch_input = []
        batch_output = []
        Pds = np.random.uniform(size=(self.batch_size))
        sample_cnt = 0
        epoch = False
        while len(batch_input) < self.batch_size:
            wrd = word_ids[self._cursor]
            
            if Pds[sample_cnt] > self.word_metadata[wrd]['Ps']:
                batch_input.append(wrd)
                targets = word_ids[self._cursor - self.window_size: self._cursor]
                epoch = self._increment_cursor(len(word_ids)) or epoch
                targets += word_ids[self._cursor: self._cursor + self.window_size]
                batch_output.append(targets)
            
            sample_cnt += 1
            if sample_cnt == self.batch_size:
                Pds = np.random.uniform(size=(self.batch_size))
                sample_cnt = 0

        return np.array(batch_input).reshape(-1, 1).astype(np.int32), np.array(batch_output), epoch
    
    def next(self):
        return self.__next__()

class PairBatchGenerator(BatchGenerator):
    def __init__(self, corpus, batch_size, split_ratio=0.8, window_size=3, loaded_data=None):
        super(PairBatchGenerator, self).__init__(corpus, batch_size, split_ratio=split_ratio, window_size=window_size, loaded_data=loaded_data)
        self._generate_pairs()
        self.shuffle()
    
    def shuffle(self):
        np.random.shuffle(self.pairs)
        
    def _generate_pairs(self):
        word_pairs = defaultdict(set)
        for idx, word in enumerate(self.corpus_words):
            wrd = self.word_to_id.get(word, None)
            if wrd is None:
                continue
            p = np.random.uniform()
            if p > self.word_metadata[wrd]['Ps']:
                lidx = max(0, idx - self.window_size)
                ridx = min(len(self.corpus_words), idx + 1 + self.window_size)
                word_pairs[wrd].update([self.word_to_id[owrd] for owrd in self.corpus_words[lidx:ridx] if owrd in self.word_to_id])
        
        pairs = []
        for wrd in word_pairs:
            for owrd in word_pairs[wrd]:
                pairs.append((wrd, owrd))
        
        self.pairs = pairs
    
    def __iter__(self):
        self._cursor = 0
        return self
    
    def __next__(self):
        if self._cursor + self.batch_size >= len(self.pairs):
            rem = (self._cursor + self.batch_size) - len(self.pairs)
            batch = np.array(self.pairs[self._cursor:] + self.pairs[:rem])
            self._cursor = rem
            return batch[:, 0].reshape(-1, 1), batch[:, 1].reshape(-1, 1), True
        _pcursor = self._cursor
        batch = np.array(self.pairs[self._cursor: self._cursor + self.batch_size])
        self._cursor = (self._cursor + self.batch_size) % len(self.pairs)
        return batch[:, 0].reshape(-1, 1), batch[:, 1].reshape(-1, 1), True if _pcursor >= self._cursor else False

    def next(self):
        return self.__next__()
