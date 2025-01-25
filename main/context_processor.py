from .models import Profile, Place


def profile_around(request):
    profile = Profile.objects.first()
    return {"profile": profile}


def places_around(request):
    places = Place.objects.all()
    return {"places": places}
