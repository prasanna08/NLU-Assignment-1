# NLU Assignment 1
This code implements the core logic of word2vec in tensorflow along with subsampling and negative sampling.

## Training model from scratch.
To train model from scratch use `word2vec.py`. Run following command:
```
python word2vec.py [--embedding_dim EMBEDDING_DIM] [--skip_window SKIP_WINDOW] [--epochs EPOCHS] [--neg_samples NEG_SAMPLES] [--batch_size BATCH_SIZE] [--summary_steps SUMMARY_STEPS] [--checkpoint_steps CHECKPOINT_STEPS]
```
The model will store checkpoint at given number of epochs and also print the loss function at given number of steps.


## Model evaluation.
To evaluate model use following command:
```
python Evaluate.py [-h] [--k K] [--eval-simlex EVAL_SIMLEX] [--word-analogy WORD_ANALOGY] [--similarity SIMILARITY] [--word-analogy-score WORD_ANALOGY_SCORE] [--closest-words CLOSEST_WORDS]
```
Arguments:
*  `--k`: Specify neighborhood size
*  `--eval-simlex True`: Evaluate simlex dataset
*  `--word-analogy 'word1 word2 word3'`: Input 3 space separated words (in quoated string), output is k closest word to word analogy
*  `--similarity`: Cosine similarity between two space separated words* 
*  `--word-analogy-score True`: Evaluate on word analogy task
*  `--closest-words word`: Returns top k closest word for given input word
