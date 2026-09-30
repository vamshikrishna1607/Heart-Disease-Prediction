from django.urls import path

from . import views

app_name = "predictor"

urlpatterns = [
    path("", views.predict, name="predict"),
    path("about/", views.about, name="about"),
]
