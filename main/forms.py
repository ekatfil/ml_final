from django import forms
from .models import UserRequest
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class UserRequestForm(forms.ModelForm):
    class Meta:
        model = UserRequest
        fields = ["detection_method", "uploaded_photo"]


class RegisterForm(UserCreationForm):
    email = forms.EmailField(
        required=True, widget=forms.EmailInput(attrs={"class": "form-control"})
    )
    username = forms.CharField(widget=forms.TextInput(attrs={"class": "form-control"}))
    password1 = forms.CharField(
        widget=forms.PasswordInput(attrs={"class": "form-control"})
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput(attrs={"class": "form-control"})
    )

    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]
