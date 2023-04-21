from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = 'topic_classification'
urlpatterns = [
    # two paths: with or without given topic
    path('', views.index, name='index'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)