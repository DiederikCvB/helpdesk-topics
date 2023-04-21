from django.shortcuts import render
from django.conf import settings
from django.templatetags.static import static

from .forms import CommentForm

from pathlib import Path
import joblib

from DiedsModule.TopicModel import Preprocessor, TopicPredictor

# Load models and preprocessing once

version = '_customers'
nmf = joblib.load(settings.STATIC_ROOT / f'nmf{version}.joblib')
vectorizer = joblib.load(settings.STATIC_ROOT / f'vectorizer{version}.joblib')

preproc = Preprocessor()
topic_model = TopicPredictor(nmf, vectorizer, version)

# Create your views here.

def index(request):
    predicted_label = None
    text = None
    descr = None
    chart= None
    if request.method == 'POST':
        # in case of POST: get the uploaded text from the form and process it
        form = CommentForm(request.POST)
        if form.is_valid():
            # retrieve the uploaded text
            text = form.cleaned_data['message']

            # clean the text
            df = preproc.clean(text)
            try: #check if we have enough material left after cleaning
                stripped = df['clean_text'].str.replace('\s+', '' , regex=True)
                assert  len(stripped.loc[0]) > 1, \
                         f"After stripping whitespaces, URL's, symbols etc. there is too little left: \n \"{stripped.loc[0]}\""
            except AssertionError as ve:
                print(ve)
                return render(
                    request,
                    'ciphix_topics/index.html',
                    {
                        'form': form,
                        'entered_text' : text,
                        'errormessage' : ve,
                    }
                )

            df = preproc.preprocess(df)

            # get predicted label
            try:
                predicted_label, descr = topic_model.predict(df)
                print(f"PREDICT: {predicted_label}")
                chart = topic_model.visualize_topic(predicted_label)
            except RuntimeError as re:
                print(re)

    else:
        # in case of GET: simply show the empty form for uploading text
        form = CommentForm()

    # pass the form, text, and predicted label to the template to be rendered
    context = {
        'form': form,
        'predicted_label': predicted_label,
        'entered_text' : text,
        'descr' : descr,
        'chart' : chart,
    }
    return render(request, 'ciphix_topics/index.html', context)