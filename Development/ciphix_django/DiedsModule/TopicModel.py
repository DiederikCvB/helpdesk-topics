import pandas as pd
import re
import spacy
import en_core_web_sm

class Preprocessor:
    def __init__(self) -> None:
        self.nlp = en_core_web_sm.load(disable=["parser", "ner", "textcat"])
        
    def preprocess(self, df):
        docs = self.nlp.pipe(df['clean_text'], n_process=2)
        processed = []
        for doc in docs:
            pos_sel = " ".join(token.lemma_ for token in doc if (token.pos_ in ['PROPN','NOUN','VERB'] and not token.is_stop))
            processed.append(pos_sel)
        df['processed_text'] = processed
        return df
    
    def clean(self, text):
        df = pd.DataFrame([text], dtype=str, columns=['text'])
        print(df)
        df['clean_text'] = df['text'].apply(self.remove_ats) \
            .apply(self.remove_tag) \
            .apply(self.remove_urls) \
            .apply(self.remove_emoji) \
            .apply(self.remove_specialchars) \
            .apply(self.replace_hashtags) 
        return df    

    def remove_ats(self, text):
    #Remove all @ tags        
        at_pattern = re.compile('@[a-zA-Z\d_]+')
        return at_pattern.sub(r'', text)
    
    def remove_tag(self, text):
    #Remove all employee tags
    #Tags occur at the end of the line with capital letters and prefix '-' or '^'           
        at_pattern = re.compile('[\^\-][A-Z\d]+$')
        return at_pattern.sub(r'', text)
    
    def remove_urls(self, text):
        #Remove URLS
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def remove_emoji(self, text):  
        #Remove smileys 
        emoji_pattern = re.compile("["
                                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                                "\U0001F600-\U0001F64F"  # emoticons
                                "\U0001F680-\U0001F6FF"  # transport & map symbols
                                "\U0001F700-\U0001F77F"  # alchemical symbols
                                "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                "\U0001FA00-\U0001FA6F"  # Chess Symbols
                                "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                "\U00002702-\U000027B0"  # Dingbats
                                "\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
  
    def remove_specialchars(self, text):
        #Remove newlines.
        char_pattern = re.compile('[\n\'\"]')
        return char_pattern.sub(r'', text)

    def replace_hashtags(self, text):
        #Replace hashtags with spaces, these could be useful words so we want to recover them
        char_pattern = re.compile('[\#]')
        return char_pattern.sub(r' ', text)
    
    
class TopicPredictor:
    def __init__(self, nmf, vectorizer, version=''):
        self.topic_count = 10
        self.nmf = nmf
        self.vectorizer = vectorizer
        self.topic_descr = self.get_topics_descr()
        print("done")

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

    def predict(self, doc):
        """
        Predicts the topic of a new piece of raw text 
        Returns: topic number and description of that topic in word scores
        """
        assert len(doc) > 0, f"number greater than 0 expected, got: {number}"
        vectorized = self.vectorizer.transform(doc["processed_text"])
        nmf = self.nmf.transform(vectorized) 
        print("Processed text:", doc['processed_text'])

        predicted_topic = [each.argsort()[::-1][0] for each in nmf]
        topic_descr = self.topic_descr[predicted_topic[0]]
        return predicted_topic, topic_descr    