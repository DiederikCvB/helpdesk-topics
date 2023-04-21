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

    if request.method == 'POST':
        # in case of POST: get the uploaded text from the form and process it
        form = CommentForm(request.POST)
        if form.is_valid():
            # retrieve the uploaded text
            text = form.cleaned_data['comment']
            print('%'*50)

            # clean the text
            df = preproc.clean(text)
            df = preproc.preprocess(df)
            print('Preprocessed: \n',df, '\n')

            # get predicted label
            try:
                predicted_label, descr = topic_model.predict(df)
                print(f"PREDICT: {predicted_label}")
            except RuntimeError as re:
                print(re)

    else:
        # in case of GET: simply show the empty form for uploading text
        form = CommentForm()

    # pass the form, image URI, and predicted label to the template to be rendered
    context = {
        'form': form,
        'predicted_label': predicted_label,
        'entered_text' : text,
        'descr' : descr,
    }
    return render(request, 'ciphix_topics/index.html', context)