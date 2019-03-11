import nltk
import batch_generators as bgen
import tensorflow as tf
import numpy as np
import argparse
import os

try:
    from nltk.corpus import reuters
except Exception:
    import nltk
    nltk.download('reuters')
finally:
    from nltk.corpus import reuters

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', help='Specify embedding dimesnion', default=256 ,type=np.int32)
parser.add_argument('--skip_window', help='Specify window size', default=3, type=np.int32)
parser.add_argument('--epochs', help='Specify maximum epochs', default=5, type=np.int32)
parser.add_argument('--neg_samples', help='Specify number of negative samples to be generated for Noise Contrastive Estimation',
                    default=8, type=np.int32)
parser.add_argument('--batch_size', help='Specify batch size', default=128, type=np.int64)
parser.add_argument('--summary_steps', help='Save summary to TensorBoard every n steps', default=50, type=np.int32)
parser.add_argument('--checkpoint_steps', help='Save checkpoints at every n steps of minibatch', default=2, type=np.int32)

def word2vec(corpus, args):
    bg = bgen.PairBatchGenerator(corpus, args.batch_size, window_size=args.skip_window)
    print(len(bg.pairs))
    n_words = len(bg.word_to_id)
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    n_neg_samples = args.neg_samples
    window_size = args.skip_window

    with tf.name_scope('Word2Vec-skipgram'):
        with tf.name_scope('Inputs'):
            inputs = tf.placeholder(dtype=tf.int64, shape=(None, 1), name='input-word')
            targets = tf.placeholder(dtype=tf.int64, shape=(None, 1), name='output-words')
            neg_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
                true_classes=targets, num_true=1, num_sampled=n_neg_samples, unique=True,
                range_max=n_words, unigrams=bg.get_negative_probs(), distortion=0.75)
        with tf.name_scope('Embeddings'):
            U = tf.Variable(tf.truncated_normal((n_words, embedding_dim)), name='U')
            V = tf.Variable(tf.truncated_normal((n_words, embedding_dim)), name='V')
        with tf.name_scope('Score'):
            E1 = tf.reshape(tf.nn.embedding_lookup(U, inputs), (-1, embedding_dim))
            E2 = tf.reshape(tf.nn.embedding_lookup(V, targets), (-1, embedding_dim))
            E3 = tf.negative(tf.nn.embedding_lookup(V, neg_samples))
        with tf.name_scope('Loss'):
            pos_loss = tf.log(tf.nn.sigmoid(tf.reduce_sum(tf.multiply(E1, E2), axis=1)))
            neg_loss = tf.reduce_sum(tf.log(tf.nn.sigmoid(tf.matmul(E1, tf.transpose(E3)))), axis=1)
            loss = tf.negative(tf.add(tf.reduce_mean(pos_loss, axis=0), tf.reduce_mean(neg_loss, axis=0)))
            loss_summary = tf.summary.scalar('loss', loss)
        with tf.name_scope('Optimizer'):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            initial_learning_rate = 0.01
            learning_rate = tf.train.exponential_decay(
                initial_learning_rate, global_step, 2000, 0.96, staircase=True)
            opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
            merged_op = tf.summary.merge([loss_summary])

    sess = tf.Session()
    # Writer has to be initialzed before all variables are initialized.
    writer = tf.summary.FileWriter('train/NCE-%d-%d-%d' % (embedding_dim, batch_size, n_neg_samples), sess.graph)
    summary_steps = args.summary_steps
    checkpoint_steps = args.checkpoint_steps
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    checkpoint_path = './checkpoint/NCE-%d-%d-%d/' % (embedding_dim, batch_size, n_neg_samples)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    max_epochs = args.epochs
    epoch = 0
    step = 0
    try:
        for (x, y, p) in bg:
            _, step_loss = sess.run([opt, loss], feed_dict={inputs:x, targets:y})
            if p:
                print("Epoch %d" % epoch)
                epoch += 1
                if epoch >= max_epochs:
                    saver.save(sess, '%s/word2vec-%d.ckpt' % (checkpoint_path, epoch))
                    break
            if step % 20 == 0:
                print("Step: %d, Loss: %.3f" % (step, step_loss))
            if step % summary_steps == 0:
                merged = sess.run([merged_op], feed_dict={inputs:x, targets:y})[0]
                writer.add_summary(merged, step)
            if epoch % checkpoint_steps == 0:
                saver.save(sess, '%s/word2vec-%d.ckpt' % (checkpoint_path, epoch))
            step += 1
    except Exception:
        saver.save(sess, '%s/word2vec-%d.ckpt' % (checkpoint_path, epoch))

    writer.close()
    bg.store_metadata('batch_generator_metadata-2.json')

if __name__ == '__main__':
    args = parser.parse_args()
    print("Starting word2vec training.")
    word2vec(reuters, args)
