from django.shortcuts import render, redirect
from .models import UserRequest, DetectionMethod
from .forms import UserRequestForm, RegisterForm
from .detection import Detector
from django.core.files.base import ContentFile
import os
from django.contrib.auth import login


def index(request):
    if request.method == "POST":
        form = UserRequestForm(request.POST, request.FILES)
        if form.is_valid():
            user_request = form.save(commit=False)
            user_request.user = request.user
            user_request.save()

            # Обработка фото
            detector = Detector()
            uploaded_photo_path = user_request.uploaded_photo.path
            processed_photo_path = os.path.join(
                "media", "processed", os.path.basename(uploaded_photo_path)
            )
            result = detector.get_prediction(
                uploaded_photo_path, model_name=user_request.detection_method.name
            )

            # Сохранение обработанного фото
            user_request.processed_photo.save(
                os.path.basename(processed_photo_path), ContentFile(result)
            )
            user_request.save()

            return redirect("history")
    else:
        form = UserRequestForm()

    context = {
        "title": "Главная страница",
        "form": form,
    }

    return render(request, "main/index.html", context)


def history(request):
    user_requests = UserRequest.objects.filter(user=request.user).order_by("-date")

    context = {
        "title": "История запросов",
        "user_requests": user_requests,
    }
    return render(request, "main/history.html", context)


def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("index")
    else:
        form = RegisterForm()

    context = {
        "title": "Регистрация",
        "form": form,
    }
    return render(request, "main/register.html", context)
