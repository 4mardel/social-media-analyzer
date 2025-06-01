import praw
import os
import json
import requests
import time
from datetime import datetime, timezone
from config import CLIENT_ID, CLIENT_SECRET, USER_AGENT

def init_reddit():
    return praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )

def download_image(url, folder, filename):
    try:
        response = requests.get(url, stream=True, timeout=5)
        if response.status_code == 200:
            with open(os.path.join(folder, filename), 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False

def crawl_subreddit(reddit, subreddit_name='Art', limit=50):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    image_folder = 'data/raw/images'
    os.makedirs(image_folder, exist_ok=True)

    count = 0
    skipped = 0

    print(f"\nCrawling subreddit: {subreddit_name} (limit: {limit})")

    for post in subreddit.hot(limit=limit):
        if any(ext in post.url.lower() for ext in ['.jpg', '.jpeg', '.png']):
            filename = f"{post.id}.jpg"
            if download_image(post.url, image_folder, filename):
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'image_path': os.path.join('images', filename),
                    'upvotes': post.score,
                    'created_utc': datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
                    'subreddit': subreddit_name,
                    'url': post.url
                }
                posts.append(post_data)
                count += 1
                print(f"Saved: {filename}")
            else:
                skipped += 1
        else:
            print(f"Skipped (not image): {post.url}")
            skipped += 1

        time.sleep(0.5)

    print(f"\nTotal collected: {count} | Skipped: {skipped}")

    out_path = f"data/raw/reddit_data_{subreddit_name.lower()}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)

    print(f"Saved {count} posts to {out_path}")


if __name__ == "__main__":
    reddit = init_reddit()

    target_subreddits = ['Art', 'pics', 'cats']

    for sub in target_subreddits:
        crawl_subreddit(reddit, subreddit_name=sub, limit=50)