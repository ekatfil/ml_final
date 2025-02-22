from django.db import models
from django.contrib.auth.models import User


class DetectionMethod(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(null=True)

    def __str__(self):
        return self.name


class UserRequest(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    detection_method = models.ForeignKey(DetectionMethod, on_delete=models.CASCADE)
    uploaded_photo = models.ImageField(upload_to="uploads/")
    processed_photo = models.ImageField(upload_to="processed/", null=True, blank=True)
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Request by {self.user.username} on {self.date}"
