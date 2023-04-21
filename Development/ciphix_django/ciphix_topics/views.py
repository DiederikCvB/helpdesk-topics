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
    image_uri = None
    predicted_label = None

    if request.method == 'POST':
        # in case of POST: get the uploaded image from the form and process it
        form = CommentForm(request.POST)
        if form.is_valid():
            # retrieve the uploaded image and convert it to bytes (for PyTorch)
            text = form.cleaned_data['comment']
            print('%'*50)

            # clean the text
            df = preproc.clean(text)
            df = preproc.preprocess(df)
            print('Preprocessed: \n',df, '\n')
            
            # get predicted label with previously implemented PyTorch function
            try:
                predicted_label, descr = topic_model.predict(df)
            except RuntimeError as re:
                print(re)

    else:
        # in case of GET: simply show the empty form for uploading images
        form = CommentForm()

    # pass the form, image URI, and predicted label to the template to be rendered
    context = {
        'form': form,
        'predicted_label': predicted_label,
        'descr' : descr,
    }
    return render(request, 'ciphix_topics/index.html', context)