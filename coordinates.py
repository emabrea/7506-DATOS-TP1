#!/usr/bin/env python3

import pandas as pd
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="notebook", timeout=300)

latitudes = []
longitudes = []
def download_coordinates(ciudad):
    data = None
    data = geolocator.geocode(ciudad)
    if data is None:
            latitudes.append(None)
            longitudes.append(None)
            return
    print(ciudad, "OK")
    latitudes.append(data.latitude)
    longitudes.append(data.longitude)

df = pd.read_csv("events.csv", usecols=["city"])
cities = df.loc[df["city"] != "Unknown"].dropna()
cities = cities["city"].value_counts().to_frame("cantidad")
cities.index.name = 'ciudad'
for city in cities.index:
    download_coordinates(city)
cities = cities.assign(latitud=latitudes)
cities = cities.assign(longitud=longitudes)

cities.to_csv("coordinates.csv")
