from newsapi import NewsApiClient
from dotenv import load_dotenv
import os
import re
from datetime import datetime
from newspaper import Article
import time
import json
from investopedia import get_investopedia_news

def get_all_news(query):
    return newsapi.get_everything(
        q = query,
        sources = 'bloomberg, the-new-york-times, \
        the-washington-post, the-wall-street-journal, politico, \
        reuters, financial-times, cnbc, techcrunch, the-verge',
        domains = 'bloomberg.com, techcrunch.com',
        to = datetime.now().strftime('%Y-%m-%d'), # today
        language = 'en',
        sort_by = 'relevancy',
)
# ----------------------------------------------  

# Function to save articles to a JSON file and return file name
def articles_dump(_q):
    # Check dump directory for recent files, vaild: 24 hrs
    now = int(time.time() * (10 ** 3))
    cutoff = now - 24*3600*1000  # 24 hours in ms
    pattern = re.compile(rf"^{re.escape(_q)}_(\d+)\.jsonl$")
    recent_files = []

    for fname in os.listdir("dump"):
        match = pattern.match(fname)
        if match:
            timestamp = int(match.group(1))
            if timestamp >= cutoff:
                return "./dump/" + fname


    load_dotenv()
    news_api_key = os.getenv("NEWSAPI_KEY")
    global newsapi
    newsapi = NewsApiClient(api_key = news_api_key)

    allNews = get_all_news(_q)

    results = []
    urls = get_investopedia_news(_q)

    for article in allNews['articles']:
        urls.append(article['url'])

    for url in urls:
        print(f"Processing: {url}")
        try:
            art = Article(url)
            art.download()
            art.parse()
            if art.authors == []:
                continue
            results.append(art)

        except Exception as e:
            print(f"Failed to process {url}: {e}")

        time.sleep(0.2)

    SAVE_PATH = os.path.join(".", "dump")
    fname = _q
    
    filename = fname + '_' + str(int(time.time() * (10 ** 3))) + '.jsonl'
    os.makedirs(SAVE_PATH, exist_ok = True)
    path = os.path.join(SAVE_PATH, filename)

    with open(path, "w", encoding = "utf-8") as f:
        for article in results:
            try:
                json_obj = {
                    "title": article.title,
                    "authors": article.authors,
                    "text": article.text,
                    "publish_date": article.publish_date.isoformat() if article.publish_date else None,
                    "source_url": article.source_url if hasattr(article, 'source_url') else None,
                    "url": article.url if hasattr(article, 'url') else None,
                }
                json.dump(json_obj, f, ensure_ascii = False)
                f.write("\n")
            except Exception as e:
                print(f"Skipping article due to error: {e}")

    print(f"Saved {len(results)} articles to {path}")
    return path
