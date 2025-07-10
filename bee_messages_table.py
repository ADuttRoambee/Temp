import requests
from tabulate import tabulate
from datetime import datetime, timedelta, timezone
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --- User Configuration ---
API_URL = "https://api-staging.roambee.com/bee/bee_messages"
API_KEY = "fe341abc-c446-4900-ade6-47ad81307de2"
BEE_ID = "868199058126286"
DAYS = 5

headers = {
    "apikey": API_KEY,
    "Content-Type": "application/json"
}
params = {
    "bid": BEE_ID,
    "days": DAYS
}

def parse_sent_time(sent_time):
    try:
        return datetime.strptime(sent_time, "%y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return sent_time

# Set up geopy with Nominatim (OpenStreetMap)
geolocator = Nominatim(user_agent="my_agent")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

def get_address(lat, lon):
    try:
        location = reverse((lat, lon), exactly_one=True)
        return location.address if location else "Address not found"
    except Exception as e:
        print(f"Error getting address: {e}")
        return "Address not found"

response = requests.get(API_URL, headers=headers, params=params)
if response.status_code != 200:
    print(f"Error: {response.status_code} - {response.reason}")
    print("Response content:", response.text)
    exit(1)

data = response.json()



print(data)


if not data:
    print("No bee messages found for the given parameters.")
else:
    table = []
    for msg in data:
        sent_time = msg.get("sent_time") or msg.get("gps_time") or ""
        latitude = msg.get("latitude") or msg.get("lat") or ""
        longitude = msg.get("longitude") or msg.get("lon") or ""
        if sent_time and latitude and longitude:
            readable_time = parse_sent_time(sent_time)
            address = get_address(latitude, longitude)
            table.append([readable_time, latitude, longitude, address])
    print(tabulate(table, headers=["Exact Time", "Latitude", "Longitude", "Address"], tablefmt="grid"))
