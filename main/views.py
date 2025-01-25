from django.shortcuts import render
from .models import Main, About, DishCategory, Dish


def index(request):
    main = Main.objects.first()
    about_items = About.objects.all()
    popular_dishes = Dish.objects.filter(category__name="Популярное")
    set_dishes = Dish.objects.filter(category__name="Сеты и пары")

    categories = DishCategory.objects.exclude(name__in=["Популярное", "Сеты и пары"])

    context = {
        "title": "Главная страница",
        "main": main,
        "about_items": about_items,
        "popular_dishes": popular_dishes,
        "set_dishes": set_dishes,
        "categories": categories,
    }
    return render(request, "main/index.html", context)


def contacts(request):
    context = {
        "title": "Контакты",
    }
    return render(request, "main/contacts.html", context)
