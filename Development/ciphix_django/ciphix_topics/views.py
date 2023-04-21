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

preproc = Preprocessor
topic_model = TopicPredictor(nmf, vectorizer, version)

# Create your views here.

def index(request):
    
    return render(request, 'ciphix_topics/index.html', context)