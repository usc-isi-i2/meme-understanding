import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import gensim
from gsdmm import MovieGroupProcess
import re
import string
from nltk.corpus import stopwords
import os
import joblib
stopwords = stopwords.words('english')


def preprocess(text: str):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords])
    text = ' '.join([word for word in text.split() if len(word) >= 3])
    text = ' '.join([word for word in text.split()
                    if not re.match(r'\b\w+\.(com|org|net)\b', word)])
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if len(word) >= 3])
    return text


def train_gsdmm(
    training_data: str or Path,
    cache_dir: str or Path,
    num_topics: int = 6,
    num_words_per_topic: int = 20,
    num_iterations: int = 15,
):
    df = pd.read_csv(training_data, delimiter='\t')
    texts = df['Text Transcription'].tolist()
    texts = list(map(preprocess, texts))

    docs = [text.split() for text in texts]

    dictionary = gensim.corpora.Dictionary(docs)

    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    vocab_length = len(dictionary)

    gsdmm = MovieGroupProcess(K=num_topics, alpha=0.1,
                              beta=0.3, n_iters=num_iterations)

    gsdmm.fit(docs, vocab_length)

    doc_count = np.array(gsdmm.cluster_doc_count)
    print('Number of documents per topic :', doc_count)

    top_index = doc_count.argsort()[-num_topics:][::-1]

    topics_words = {}
    for topic_num in top_index:
        topic_dict = dict(sorted(gsdmm.cluster_word_distribution[topic_num].items(
        ), key=lambda k: k[1], reverse=True)[:num_words_per_topic])
        topics_words[topic_num] = list(topic_dict.keys())

    joblib.dump(gsdmm, os.path.join(cache_dir, 'gsdmm.pkl'))
    joblib.dump(topics_words, os.path.join(cache_dir, 'topics_keywords.pkl'))


def predict_label(
    gsdmm: MovieGroupProcess,
    topics_words: Dict[str, Any],
    text: str,
):
    doc = preprocess(text).split()
    topic, confidence = gsdmm.choose_best_label(doc)
    return {
        'topic': topic,
        'topic_words': topics_words[topic],
        'confidence': confidence,
    }


def read_text(path: str or Path):
    with open(path, 'r') as f:
        texts = f.read().splitlines()
    return texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_to_process', required=True,
                        help='Data to process', type=str)
    parser.add_argument('--cache_dir', required=True, type=str)
    parser.add_argument('--task', type=str,
                        help="Task to perform", choices=['train', 'predict'])

    args = parser.parse_args()

    if args.task == 'train':
        train_gsdmm(
            training_data=args.data_to_process,
            cache_dir=args.cache_dir,
        )

    elif args.task == 'predict':
        gsdmm = joblib.load(os.path.join(args.cache_dir, 'gsdmm.pkl'))
        topics_words = joblib.load(os.path.join(
            args.cache_dir, 'topics_keywords.pkl'))
        texts = read_text(args.data_to_process)
        predictions = [predict_label(gsdmm, topics_words, text)
                       for text in texts]
        joblib.dump(predictions, os.path.join(
            args.cache_dir, 'predictions.pkl'))
        # return predict_label(gsdmm, topics_words, args.data_to_process)
