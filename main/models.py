from django.db import models


class Profile(models.Model):
    name = models.CharField(max_length=100)
    logo = models.FileField(upload_to="uploads/")
    map = models.TextField()

    class Meta:
        verbose_name = "Профиль компании"
        verbose_name_plural = "Профиль компании"

    def __str__(self):
        return self.name


class Main(models.Model):
    title = models.TextField()
    description = models.TextField()

    class Meta:
        verbose_name = "Главный блок"
        verbose_name_plural = "Главный блок"

    def __str__(self):
        return self.title


class About(models.Model):
    title = models.CharField(max_length=100)
    description = models.TextField()

    class Meta:
        verbose_name = "О компании"
        verbose_name_plural = "О компании"

    def __str__(self):
        return self.title


class DishCategory(models.Model):
    name = models.CharField(max_length=100)

    class Meta:
        verbose_name = "Категория блюд"
        verbose_name_plural = "Категории блюд"

    def __str__(self):
        return self.name


class Dish(models.Model):
    category = models.ForeignKey(DishCategory, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    description = models.TextField(null=True, blank=True)
    price = models.CharField(max_length=100)
    img = models.FileField(upload_to="uploads/", null=True, blank=True)
    present = models.CharField(max_length=100, null=True, blank=True)

    class Meta:
        verbose_name = "Блюдо"
        verbose_name_plural = "Блюда"

    def __str__(self):
        return f"{self.category.name}: {self.name} - {self.price}₽"


class Place(models.Model):
    address = models.CharField(max_length=100)
    phone = models.CharField(max_length=100)

    class Meta:
        verbose_name = "Место"
        verbose_name_plural = "Места"

    def __str__(self):
        return f"{self.address} {self.phone}"
