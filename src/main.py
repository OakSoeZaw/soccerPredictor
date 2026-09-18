import requests

from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("FOOTBALL_API_KEY")
api_host = os.getenv("FOOTBALL_API_HOST")


def search_player(playerName):
    url = "https://free-api-live-football-data.p.rapidapi.com/football-players-search"
    querystring = {"search": playerName}

    headers = {
        "x-rapidapi-host": api_host,
        "x-rapidapi-key": api_key,
        "Content-Type": "application/json",
    }
    response = requests.get(url, headers=headers, params=querystring)
    return response.json()

print(search_player("messi"))
