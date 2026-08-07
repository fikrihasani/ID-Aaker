"""
ID-Aaker Data Collection Script (Full-Archive Search)
======================================================
Reproducible data collection using the official X API v2 full-archive search.

This script collects original posts and replies from 24 Indonesian brand accounts
across 5 Aaker brand personality dimensions. It uses the `GET /2/tweets/search/all`
endpoint. The full-archive endpoint can retrieve tweets dating back to
the first post on the platform.

Endpoint:  GET /2/tweets/search/all
Query:     from:{username} -is:retweet
Auth:      Bearer Token (OAuth 2.0 App-Only)
Max/page:  500 tweets
Pagination: pagination_token (next_token) + time-window chunking for accounts
            with very large post histories.

Requirements:
    pip install tweepy python-dotenv

Authentication:
    Set the following environment variables (or create a .env file):
        X_BEARER_TOKEN   - Your X API v2 Bearer Token (from developer.x.com)

    NOTE: The full-archive search endpoint requires at minimum a Pro-tier
    X API subscription. Pay-per-use pricing applies (~$0.005 per post read).
    Set your monthly spend cap at https://developer.x.com/en/portal/dashboard.

Tier Requirements & Cost Estimate:
    - Endpoint: GET /2/tweets/search/all (Pro tier and above)
    - Pricing: ~$0.005 per tweet read (pay-per-use model)
    - Estimated one-time cost to reproduce full dataset: ~$379 (75,756 tweets)
    - Monthly read cap: 2,000,000 posts (well above 75K)

Usage:
    python collect_id_aaker.py

Output:
    id_aaker_raw.csv  - Raw collected tweets before preprocessing (14 columns)
"""

import csv
import os
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import tweepy

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

# 24 target brand accounts with their Aaker personality assignment.
# Username → personality mapping based on expert recommendation from a
# communications researcher (PhD in Mass Communication).
BRAND_ACCOUNTS: dict[str, str] = {
    # Competence – reliable, intelligent, authoritative
    "BANKBRI_ID":        "competence",
    "HaloBCA":           "competence",
    "KAI121":            "competence",
    "TelkomIndonesia":   "competence",
    "samsungID":         "competence",

    # Excitement – bold, enthusiastic, imaginative
    "JanjiJiwaID":       "excitement",
    "Kopikenangan_ID":   "excitement",
    "erigostore":        "excitement",
    "gojekindonesia":    "excitement",
    "tokopedia":         "excitement",

    # Sophistication – upper class, charming, elegant
    "ROLEX":             "sophistication",
    "Vespa_IND":         "sophistication",
    "fsjakarta":         "sophistication",
    "s_djojokartiko":    "sophistication",
    "sociolla_id":       "sophistication",

    # Ruggedness – tough, strong, outdoorsy
    "ConsinaOfficial":   "ruggedness",
    "RoyalEnfieldID":    "ruggedness",
    "brodostoresub":     "ruggedness",
    "eigeradventure":    "ruggedness",
    "revoltindustry":    "ruggedness",

    # Sincerity – down-to-earth, honest, friendly
    "McDonalds_ID":      "sincerity",
    "NestleIndonesia":   "sincerity",
    "eminacosmetics":    "sincerity",
    "wardahbeauty":      "sincerity",
}

# Search window parameters for time-based chunking.
# Accounts with very large post histories (e.g., >50K tweets) may need
# narrower time windows to avoid hitting the per-request result cap (500).
# The script auto-narrows windows when a chunk returns exactly 500 results.
DEFAULT_WINDOW_DAYS = 180    # 6-month windows (will auto-narrow if needed)
MIN_WINDOW_DAYS     = 7      # Don't go narrower than 1 week
MAX_RESULTS         = 500    # X API v2 search/all maximum per page

# X API v2 full-archive search rate limits (Pro tier):
#   300 requests per 15-minute window, 1 request per second sustained.
RATE_LIMIT_REQUESTS = 300
RATE_LIMIT_WINDOW   = 15 * 60

# Output
# TODO: Update to your desired output directory
OUTPUT_DIR  = Path(__file__).resolve().parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "id_aaker_raw.csv"

# CSV columns for raw data (14 columns — before cleaning)
CSV_COLUMNS = [
    "tweet_id",
    "author_username",
    "author_id",
    "text",
    "created_at",
    "lang",
    "conversation_id",
    "in_reply_to_user_id",
    "favorite_count",
    "retweet_count",
    "reply_count",
    "quote_count",
    "personality",
    "is_reply",
]

# Tweet fields requested from X API v2
TWEET_FIELDS = [
    "created_at",
    "lang",
    "public_metrics",
    "conversation_id",
    "in_reply_to_user_id",
    "referenced_tweets",
    "author_id",
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("id_aaker")


# ──────────────────────────────────────────────────────────────────────
# X API Client
# ──────────────────────────────────────────────────────────────────────

def get_client() -> tweepy.Client:
    """Create an authenticated tweepy Client using Bearer Token.

    The Bearer Token (App-Only OAuth 2.0) is sufficient for reading
    public tweets from the full-archive search endpoint.

    Returns:
        tweepy.Client: Authenticated client.
    """
    bearer_token = os.getenv("X_BEARER_TOKEN")
    if not bearer_token:
        raise RuntimeError(
            "X_BEARER_TOKEN environment variable not set.\n"
            "Get your Bearer Token from https://developer.x.com/en/portal/dashboard\n"
            "Full-archive search requires Pro tier or above."
        )
    return tweepy.Client(
        bearer_token=bearer_token,
        wait_on_rate_limit=True,
        return_type=dict,
    )


# ──────────────────────────────────────────────────────────────────────
# User ID Resolution
# ──────────────────────────────────────────────────────────────────────

def resolve_user_ids(
    client: tweepy.Client,
    usernames: list[str],
) -> dict[str, str]:
    """Resolve X usernames to numeric user IDs.

    Args:
        client: Authenticated tweepy Client.
        usernames: List of usernames (without @).

    Returns:
        Dict mapping username → user_id string.
    """
    mapping: dict[str, str] = {}
    batch_size = 100
    for i in range(0, len(usernames), batch_size):
        batch = usernames[i : i + batch_size]
        log.info("Resolving %d usernames ...", len(batch))
        resp = client.get_users(usernames=batch)
        if resp.get("data"):
            for user in resp["data"]:
                mapping[user["username"]] = user["id"]
        if resp.get("errors"):
            for err in resp["errors"]:
                log.warning("  Could not resolve '%s': %s",
                            err.get("value"), err.get("detail"))
        time.sleep(1)
    log.info("Resolved %d / %d usernames.", len(mapping), len(usernames))
    return mapping


# ──────────────────────────────────────────────────────────────────────
# Full-Archive Search per Account
# ──────────────────────────────────────────────────────────────────────

def _write_tweet(tweet: dict, username: str, user_id: str,
                 personality: str, writer: csv.DictWriter) -> None:
    """Write a single tweet dict to the CSV."""
    metrics = tweet.get("public_metrics", {})
    row = {
        "tweet_id":            tweet["id"],
        "author_username":     username,
        "author_id":           user_id,
        "text":                tweet["text"],
        "created_at":          tweet.get("created_at", ""),
        "lang":                tweet.get("lang", ""),
        "conversation_id":     tweet.get("conversation_id", ""),
        "in_reply_to_user_id": tweet.get("in_reply_to_user_id", ""),
        "favorite_count":      metrics.get("like_count", 0),
        "retweet_count":       metrics.get("retweet_count", 0),
        "reply_count":         metrics.get("reply_count", 0),
        "quote_count":         metrics.get("quote_count", 0),
        "personality":         personality,
        "is_reply":            bool(tweet.get("in_reply_to_user_id")),
    }
    writer.writerow(row)


def fetch_account_tweets(
    client: tweepy.Client,
    username: str,
    user_id: str,
    personality: str,
    writer: csv.DictWriter,
    output_fh,
    stats: dict,
) -> int:
    """Fetch ALL tweets for a single account via full-archive search.

    Uses time-window chunking + per-window pagination to retrieve the
    complete tweet history. The query `from:{username} -is:retweet`
    excludes retweets while keeping original posts and replies.

    Args:
        client:      Authenticated tweepy Client.
        username:    X username.
        user_id:     X numeric user ID.
        personality: Aaker personality label.
        writer:      CSV DictWriter.
        output_fh:   File handle for periodic flush.
        stats:       Mutable stats dict.

    Returns:
        Number of tweets collected.
    """
    count = 0
    seen_ids: set[str] = set()

    # Search window: from 2006-03-21 (first tweet) to 2024-12-31
    window_end   = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    window_start = datetime(2006,  3, 21,  0,  0,  0, tzinfo=timezone.utc)

    log.info("Fetching @%s [%s] via full-archive search ...", username, personality)

    # Iterate backwards through time windows
    current_end = window_end
    while current_end > window_start:
        current_start = current_end - timedelta(days=DEFAULT_WINDOW_DAYS)
        if current_start < window_start:
            current_start = window_start

        query = f"from:{username} -is:retweet"
        pagination_token: Optional[str] = None

        chunk_count = 0
        while True:
            try:
                resp = client.search_all_tweets(
                    query=query,
                    max_results=MAX_RESULTS,
                    start_time=current_start.isoformat(),
                    end_time=current_end.isoformat(),
                    pagination_token=pagination_token,
                    tweet_fields=TWEET_FIELDS,
                )
            except tweepy.TooManyRequests:
                log.warning("  Rate limit. Sleeping 15 min ...")
                time.sleep(RATE_LIMIT_WINDOW)
                continue
            except tweepy.TweepyException as e:
                log.error("  API error for @%s: %s", username, e)
                break

            tweets = resp.get("data")
            if not tweets:
                break

            for tweet in tweets:
                tid = tweet["id"]
                if tid in seen_ids:
                    continue
                seen_ids.add(tid)

                # Safety: skip retweets that might have slipped through the query
                refs = tweet.get("referenced_tweets", [])
                ref_types = [r.get("type") for r in refs] if refs else []
                if "retweeted" in ref_types:
                    continue

                _write_tweet(tweet, username, user_id, personality, writer)
                chunk_count += 1

            # Check if this window has more pages
            meta = resp.get("meta", {})
            next_token = meta.get("next_token")
            result_count = meta.get("result_count", 0)

            if not next_token:
                break

            # If window returned exactly MAX_RESULTS and has more pages,
            # the window may be too wide — but proceed with pagination
            pagination_token = next_token
            time.sleep(0.5)  # pacing

        count += chunk_count
        if chunk_count > 0:
            log.info("  Window %s → %s: %d tweets (total: %d)",
                     current_start.strftime("%Y-%m-%d"),
                     current_end.strftime("%Y-%m-%d"),
                     chunk_count, count)

        # Move to previous time window
        current_end = current_start

        # Flush periodically
        if count % 5000 == 0:
            output_fh.flush()

        time.sleep(0.5)  # pacing between windows

    stats[personality] = stats.get(personality, 0) + count
    log.info("  ✓ @%s: %d tweets total.", username, count)
    return count


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("ID-Aaker Data Collection via X API v2 Full-Archive Search")
    log.info("=" * 60)
    log.info("Endpoint:  GET /2/tweets/search/all")
    log.info("Query:     from:{username} -is:retweet")
    log.info("Tier:      Pro (pay-per-use ~$0.005/post read)")
    log.info("Est. cost: ~$379 for full 75K dataset reproduction")
    log.info("=" * 60)

    # 1. Authenticate
    log.info("Authenticating with X API v2 (Bearer Token) ...")
    client = get_client()
    log.info("Authenticated.")

    # 2. Resolve usernames → user IDs
    usernames = list(BRAND_ACCOUNTS.keys())
    user_id_map = resolve_user_ids(client, usernames)

    missing = set(usernames) - set(user_id_map.keys())
    if missing:
        log.warning("Unresolved usernames (will skip): %s", missing)

    # 3. Open output CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_fh = open(OUTPUT_FILE, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(output_fh, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    # 4. Collect tweets per account
    stats: dict[str, int] = {}
    grand_total = 0

    for username, personality in BRAND_ACCOUNTS.items():
        user_id = user_id_map.get(username)
        if not user_id:
            log.warning("Skipping @%s — no user ID resolved.", username)
            continue

        n = fetch_account_tweets(
            client=client,
            username=username,
            user_id=user_id,
            personality=personality,
            writer=writer,
            output_fh=output_fh,
            stats=stats,
        )
        grand_total += n

    output_fh.close()

    # 5. Summary
    log.info("=" * 60)
    log.info("Collection complete.")
    log.info("Total tweets: %d", grand_total)
    log.info("Per-personality:")
    for p in ["competence", "sophistication", "excitement", "sincerity", "ruggedness"]:
        log.info("  %-16s: %d", p, stats.get(p, 0))
    log.info("Output: %s", OUTPUT_FILE)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
