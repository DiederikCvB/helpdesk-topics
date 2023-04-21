from django.shortcuts import render
from django.conf import settings

import os
import joblib
import pandas as pd


# Load models and preprocessing once
class TopicPredictor:
    def __init__(self, settings, version=''):
        self.clean_df = joblib.load(os.path.join(settings.STATIC_ROOT,f'clean_df{version}.joblib'))
        self.preprocess = joblib.load(os.path.join(settings.STATIC_ROOT,f'preprocess{version}.joblib'))

        self.nmf = joblib.load(os.path.join(settings.STATIC_ROOT,f'nmf{version}.joblib'))
        self.vectorizer = joblib.load(os.path.join(settings.STATIC_ROOT,f'vectorizer{version}.joblib'))
        self.topic_descr = self.get_topics_descr()


    def get_topics_descr(self):
        """
        Get the description of the top 10 topics in word scores
        """
        res = []
        for idx, topic in enumerate(self.nmf.components_):
            #the final {topic_count} values in reverse
            descr = [(self.vectorizer.get_feature_names_out()[i], topic[i])
                            for i in topic.argsort()[:-self.topic_count - 1:-1]] 
            res.append(descr)
        return res

    def predict(self, raw_text):
        """
        Predicts the topic of a new piece of raw text 
        Returns: topic number and description of that topic in word scores
        """
        doc = pd.DataFrame([raw_text], dtype=str, columns=['text'])

        doc['clean_text'] = self.clean_df(doc['text'])
        doc['processed_text'] = self.preprocess(doc['clean_text'])

        vectorized = self.vectorizer.transform(doc["processed_text"])
        nmf = self.nmf.transform(vectorized) 
        print("Processed text:", doc['processed_text'])

        predicted_topic = [each.argsort()[::-1][0] for each in nmf]
        topic_descr = self.topic_descr[predicted_topic[0]]
        return predicted_topic, topic_descr

topic_predictor = TopicPredictor(settings, '_customers')
# Create your views here.
