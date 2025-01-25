from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("contants/", views.contacts, name="contacts"),
]
