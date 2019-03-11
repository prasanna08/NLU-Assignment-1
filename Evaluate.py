import tensorflow as tf
import numpy as np
from batch_generators import BatchGenerator
from scipy.stats import spearmanr
import argparse

try:
    from nltk.corpus import reuters
except Exception:
    import nltk
    nltk.download('reuters')
finally:
    from nltk.corpus import reuters

parser = argparse.ArgumentParser()
parser.add_argument('--k', help='Specify neighborhood size', default=10 ,type=np.int32)
parser.add_argument('--eval-simlex', help='Evaluate simlex dataset', default=False, type=np.bool)
parser.add_argument('--word-analogy', help='Input 3 space separated words, output is k closest word to word analogy', default=None, type=str)
parser.add_argument('--similarity', help='Cosine similarity between two space separated words', default=None, type=str)
parser.add_argument('--word-analogy-score', help='Evaluate on word analogy task', default=False, type=np.bool)
parser.add_argument('--closest-words', help='Returns top k closest word for given input word', default=None, type=str)

bg = BatchGenerator.load_from_metadata('bg-data-5.json', reuters, batch_size=128)
id_to_word = dict(zip(bg.word_to_id.values(), bg.word_to_id.keys()))

def get_embeddings():
    g = tf.Graph()
    sess = tf.Session(graph=g)

    best_model = './model/word2vec.ckpt'
    saver = tf.train.import_meta_graph('%s.meta' % (best_model), graph=g)

    # Restore checkpoint
    saver.restore(sess, best_model)

    # Get embeddings matrix.
    graph = sess.graph
    U = graph.get_tensor_by_name('Word2Vec-skipgram/Embeddings/U:0')
    Wu = np.array(sess.run([U])[0])
    Wu = Wu / np.linalg.norm(Wu, axis=1).reshape(-1, 1)
    sess.close()
    return Wu


def eval_simlex(Wu):
    f = open('SimLex-999.txt', 'r')
    pairs = [[line.split('\t')[0], line.split('\t')[1], float(line.split('\t')[3])] for line in f.readlines()[1:]]
    f.close()

    get_idx = lambda x: bg.word_to_id[x] if x in bg.word_to_id else 0
    pair_ids = []
    for p in pairs:
        if p[0] in bg.word_to_id and p[1] in bg.word_to_id:
            pair_ids.append([get_idx(p[0]), get_idx(p[1]), p[2]])

    word_to_vec_scores = []
    for p in pair_ids:
        word_to_vec_scores.append([np.dot(Wu[p[0]], Wu[p[1]]), p[2]])

    word_to_vec_scores = np.array(word_to_vec_scores).astype(np.float32)
    x = word_to_vec_scores[:,0]
    y = word_to_vec_scores[:,1]

    result = spearmanr(x, y)
    print('Correlation: %.7f' % (result.correlation))

def top_k_closest_words(Wu, wrd, k=8):
    if wrd not in bg.word_to_id:
        print("Word %s is not present in vocabulary" % (wrd))
        return
    idx = bg.word_to_id[wrd]
    emb = Wu[idx]
    idcs = np.argsort(np.abs(np.dot(emb, Wu.T)))[-k:]
    return [id_to_word[i] for i in idcs[::-1]]

def word_analogy(Wu, wrd1, wrd2, wrd3, k=10):
    if wrd1 not in bg.word_to_id:
        print("Word %s is not present in vocabulary" % (wrd1))
        return
    if wrd2 not in bg.word_to_id:
        print("Word %s is not present in vocabulary" % (wrd2))
        return
    if wrd3 not in bg.word_to_id:
        print("Word %s is not present in vocabulary" % (wrd3))
        return

    idx1 = bg.word_to_id[wrd1]
    idx2 = bg.word_to_id[wrd2]
    idx3 = bg.word_to_id[wrd3]
    emb = Wu[idx1] - Wu[idx2] + Wu[idx3]
    idcs = np.argsort(np.abs(np.dot(emb, Wu.T)))[-k:]
    return [id_to_word[i] for i in idcs[::-1]]

def sim(Wu, wrd1, wrd2):
    if wrd1 not in bg.word_to_id:
        print("Word %s is not present in vocabulary" % (wrd1))
        return
    if wrd2 not in bg.word_to_id:
        print("Word %s is not present in vocabulary" % (wrd2))
        return

    idx1 = bg.word_to_id[wrd1]
    idx2 = bg.word_to_id[wrd2]
    return np.abs(np.dot(Wu[idx1], Wu[idx2]))

def word_analogy_accuracy(Wu, k=10):
    f = open('questions-words.txt', 'r')
    data = [line.split() for line in f.readlines() if not line.startswith(':')]
    correct = 0
    cnt = 0
    for d in data:
        if all([d[0].lower() in bg.word_to_id, d[1].lower() in bg.word_to_id, d[2].lower() in bg.word_to_id, d[3].lower() in bg.word_to_id]):
            cnt += 1
            if d[3].lower() in word_analogy(Wu, d[0].lower(), d[1].lower(), d[2].lower(), k=k):
                correct +=1
    print("Word analogy accuracy: %.3f" % float(correct) / cnt)

def print_words(words):
    if words is not None:
        print(', '.join(words))

if __name__ == '__main__':
    args = parser.parse_args()
    Wu = get_embeddings()
    k = args.k
    if args.eval_simlex:
        eval_simlex(Wu)

    if args.word_analogy is not None:
        wrd1, wrd2, wrd3 = [w.lower() for w in args.word_analogy.split(' ')]
        print_words(word_analogy(Wu, wrd1, wrd2, wrd3, k=k))

    if args.similarity is not None:
        wrd1, wrd2 = [w.lower() for w in args.similarity.split()]
        cosine_sim = sim(Wu, wrd1, wrd2)
        print('Cosine similarity: %.6f' % cosine_sim)

    if args.word_analogy_score:
        word_analogy_accuracy(Wu, k=k)

    if args.closest_words:
        word = args.closest_words.lower()
        print_words(top_k_closest_words(Wu, word, k=k))
