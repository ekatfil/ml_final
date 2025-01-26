from django.shortcuts import render, redirect
from .models import UserRequest, DetectionMethod
from .forms import UserRequestForm, RegisterForm
from .detection import Detector
from django.core.files.base import ContentFile
import os
import cv2
from django.contrib.auth import login


def index(request):
    detection_methods = DetectionMethod.objects.all()
    if request.method == "POST":
        form = UserRequestForm(request.POST, request.FILES)
        if form.is_valid():
            user_request = form.save(commit=False)
            user_request.user = request.user
            user_request.save()

            try:
                detector = Detector()
                uploaded_photo_path = user_request.uploaded_photo.path
                img = cv2.imread(uploaded_photo_path)
                if img is None:
                    raise ValueError("Failed to load image")

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = detector.get_prediction(
                    img, model_name=user_request.detection_method.name
                )

                if result is None:
                    raise ValueError("Detection failed")

                success, buffer = cv2.imencode(".jpg", result)
                if not success:
                    raise ValueError("Failed to encode image")

                filename = f"processed_{user_request.id}_{os.path.basename(uploaded_photo_path)}"
                user_request.processed_photo.save(
                    filename, ContentFile(buffer.tobytes()), save=True
                )
                user_request.refresh_from_db()
                if not user_request.processed_photo:
                    raise ValueError("Failed to save processed photo")

                return redirect("history")

            except Exception as e:
                user_request.delete()
                form.add_error(None, f"Processing failed: {str(e)}")
    else:
        form = UserRequestForm()

    context = {
        "title": "Главная страница",
        "form": form,
        "detection_methods": detection_methods,
        "index": True,
    }

    return render(request, "main/index.html", context)


def history(request):
    user_requests = UserRequest.objects.filter(user=request.user).order_by("-date")

    context = {
        "title": "История запросов",
        "user_requests": user_requests,
        "history": True,
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


def methods(request):
    detection_methods = DetectionMethod.objects.all()
    context = {
        "title": "Методы обработки",
        "detection_methods": detection_methods,
        "methods": True,
    }
    return render(request, "main/methods.html", context)
