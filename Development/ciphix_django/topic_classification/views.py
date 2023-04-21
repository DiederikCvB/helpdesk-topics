from django.shortcuts import render

import joblib
import pandas as pd

# Load models and preprocessing once
class TopicPredictor:
    def __init__(self, version):
        self.clean_df = joblib.load((os.path.join(settings.STATIC_ROOT,f'clean_df{version}.joblib'))
        self.preprocess = joblib.load((os.path.join(settings.STATIC_ROOT,f'preprocess{version}.joblib'))

        self.nmf = joblib.load((os.path.join(settings.STATIC_ROOT,f'nmf{version}.joblib'))
        self.vectorizer = joblib.load((os.path.join(settings.STATIC_ROOT,f'vectorizer{version}.joblib'))
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

    def predict(raw_text):
        """
        Predicts the topic of a new piece of raw text. 
        Returns: topic number and description of that topic in word scores
        """
        inf_doc = pd.DataFrame([raw_text], dtype=str, columns=['text'])

        inf_doc['clean_text'] = inf_clean_df(inf_doc['text'])
        inf_doc['processed_text'] = inf_preprocess(inf_doc['clean_text'])

        newdata_vectorized = vectorizer.transform(inf_doc["processed_text"])
        newdata_nmf = nmf.transform(newdata_vectorized) 
        print("Processed text:", inf_doc['processed_text'])

        predicted_topic = [each.argsort()[::-1][0] for each in newdata_nmf]
        topic_descr = self.topic_descr[predicted_topic[0]]
        return predicted_topic, topic_descr

TopicPredictor(version = '_customers')
# Create your views here.
