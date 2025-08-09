import requests
from bs4 import BeautifulSoup

def get_investopedia_news(query):
    
    search_url = f"https://www.investopedia.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    links = []
    for a in soup.find_all("a", href = True):
        href = a["href"]
        if query in href and href.startswith("https://"):
            links.append(href)

    return links

