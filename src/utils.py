import re

def extract_room_info(description):
    guests = re.search(r'(\d+) guests', description)
    bedrooms = re.search(r'(\d+) bedroom', description)
    beds = re.search(r'(\d+) bed', description)
    bathrooms = re.search(r'(\d+) bathroom', description)

    # Ak nenájdem hodnotu, nastavím ju na 0
    guests = int(guests.group(1)) if guests else 0
    bedrooms = int(bedrooms.group(1)) if bedrooms else 0
    beds = int(beds.group(1)) if beds else 0
    bathrooms = int(bathrooms.group(1)) if bathrooms else 0

    return guests, bedrooms, beds, bathrooms