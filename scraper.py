"""
EV Sentiment — Scraper Pipeline
================================
Sources : 91Wheels · Bikewale · Bikedekho
Models  : TVS iQube · Ather Rizta · Ola S1X · Bajaj Chetak · Hero Vida V2

Run     : python scraper.py
Schedule: every 2 weeks via cron —  0 2 1,15 * *  python /path/scraper.py
Output  : data/master_reviews.csv  (cumulative, deduped)
          data/batches/<YYYYMMDD>_batch.csv  (each run's net-new rows)
"""

import re
import json
import time
import hashlib
from logger import log
import requests
import cloudscraper
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta
from bs4 import BeautifulSoup


# # ── Logging ──────────────────────────────────────────────────────────────────
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s  %(levelname)-7s  %(message)s",
# )


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
BATCH_DIR   = DATA_DIR / "batches"
MASTER_PATH = DATA_DIR / "master_reviews.csv"

DATA_DIR.mkdir(exist_ok=True)
BATCH_DIR.mkdir(exist_ok=True)

# ── Schema ────────────────────────────────────────────────────────────────────
# Only these columns are kept in the final dataset
FINAL_COLS = ["review_id", "brand", "model", "source",
              "user_name", "review_text", "rating",
              "posted_date", "scraped_at"]


# ═════════════════════════════════════════════════════════════════════════════
class Scraper:

    # ── Config ────────────────────────────────────────────────────────────────
    #
    # urls[model] = [91wheels_url, bikewale_url, bikedekho_url]
    # Use {i} as the page placeholder.
    #
    MODELS = {
        "TVS iQube": [
            "https://www.91wheels.com/scooters/tvs/iqube-electric/reviews/page{i}",
            "https://www.bikewale.com/tvs-bikes/iqube/reviews/page/{i}",
            "https://www.bikedekho.com/tvs/iqube-electric/reviews?pageno={i}",
        ],
        "Ather Rizta": [
            "https://www.91wheels.com/scooters/ather/rizta/reviews/page{i}",
            "https://www.bikewale.com/ather-bikes/rizta/reviews/page/{i}",
            "https://www.bikedekho.com/ather-energy/rizta/reviews?pageno={i}",
        ],
        "Ola S1X": [
            "https://www.91wheels.com/scooters/ola-electric/s1-x-gen-3/reviews/page{i}",
            "https://www.bikewale.com/ola-bikes/s1-x/reviews/page/{i}",
            "https://www.bikedekho.com/ola-electric/s1x/reviews?pageno={i}",
        ],
        "Bajaj Chetak": [
            "https://www.91wheels.com/scooters/bajaj/chetak/reviews/page{i}",
            "https://www.bikewale.com/bajaj-bikes/chetak/reviews/page/{i}",
            "https://www.bikedekho.com/bajaj/chetak/reviews?pageno={i}",
        ],
        "Hero Vida V2": [
            "https://www.91wheels.com/scooters/hero-vida/vida-2/reviews/page{i}",
            "https://www.bikewale.com/vida-bikes/v2/reviews/page/{i}",
            "https://www.bikedekho.com/vida/vida-v2/reviews?pageno={i}",
        ],
    }

    # Brand lookup (derived from model name)
    BRAND_MAP = {
        "TVS iQube":   "TVS",
        "Ather Rizta": "Ather",
        "Ola S1X":     "Ola Electric",
        "Bajaj Chetak":"Bajaj",
        "Hero Vida V2":"Hero",
    }

    # Max pages to scrape per (model, source) — tune per site
    MAX_PAGES = {
        "91wheels":  40,
        "bikewale":  25,
        "bikedekho": 18,
    }

    # 91Wheels CSS selectors
    WHEELS91_TAGS = {
        "user_name":   "span.text-sm.text-darkblack\\/80",
        "rating":      "span.text-xs.font-medium.text-gray-700",
        "posted_date": "span.text-xs.text-darkblack\\/60.flex-shrink-0",
        "review_text": "div.text-sm.text-gray-700.leading-relaxed",
    }

    # ── Init ──────────────────────────────────────────────────────────────────
    def __init__(self):
        self.scraped_at = datetime.now().strftime("%d-%m-%Y")
        # BUG FIX: store as datetime object, not string, for timedelta math
        self._now = datetime.now()
        self._cloud = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True}
        )

    # ── Date helpers ──────────────────────────────────────────────────────────

    def _relative_to_date(self, date_str: str) -> str:
        """
        Converts relative strings like '2 weeks ago', '1 year ago', '3 days ago'
        into DD-MM-YYYY.
        """
        parts = date_str.lower().strip().split()
        try:
            n = int(parts[0])
            unit = parts[1] if len(parts) > 1 else ""
            if "week"  in unit: delta = timedelta(weeks=n)
            elif "year" in unit: delta = timedelta(days=365 * n)
            elif "day"  in unit: delta = timedelta(days=n)
            elif "month" in unit: delta = timedelta(days=30 * n)
            else:                 delta = timedelta(0)
            return (self._now - delta).strftime("%d-%m-%Y")
        except (ValueError, IndexError):
            return self._now.strftime("%d-%m-%Y")

    @staticmethod
    def _bikedekho_date(date_str: str) -> str:
        """Convert 'Nov 25, 2025' → '25-11-2025'."""
        # BUG FIX: 'return' keyword removed from lambda — put logic here
        try:
            return datetime.strptime(date_str, "%b %d, %Y").strftime("%d-%m-%Y")
        except ValueError:
            return date_str
        # return self._now.strftime("%d-%m-%Y")

    @staticmethod
    def _make_review_id(source: str, user_name: str, review_text: str) -> str:
        """Stable unique ID per review — used for deduplication across runs."""
        raw = f"{source}::{user_name.lower().strip()}::{review_text[:100].strip()}"
        return hashlib.md5(raw.encode()).hexdigest()

    # ── 91Wheels scraper ──────────────────────────────────────────────────────

    def _scrape_91wheels(self, url_template: str, model: str) -> pd.DataFrame:

        rows = []
        source = "91wheels"
        max_pages = self.MAX_PAGES[source]

        for i in range(1, max_pages + 1):
            url = url_template.format(i=i)       
            log.info("  [91wheels] %s page %d", model, i)
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    log.warning("  HTTP %s — stopping", resp.status_code)
                    break
                soup = BeautifulSoup(resp.content, "html.parser")
            except requests.RequestException as e:
                log.warning("  Request error: %s", e)
                break

            cards = soup.find_all(
                "div",
                class_="border rounded-lg p-3 transition-all bg-white border-theme-primary/20"
            )
            if not cards:
                log.info("  No cards on page %d — end of results", i)
                break

            for card in cards:
                row = {}
                for key, selector in self.WHEELS91_TAGS.items():
                    el = card.select_one(selector)
                    row[key] = el.get_text(strip=True) if el else None
                rows.append(row)

            time.sleep(1.5)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["posted_date"] = df["posted_date"].apply(
            lambda x: self._relative_to_date(x) if x else self.scraped_at
        )
        df["source"]     = source
        df["model"]      = model
        df["brand"]      = self.BRAND_MAP[model]
        df["scraped_at"] = self.scraped_at
        df["rating"]     = pd.to_numeric(df["rating"], errors="coerce")
        return df

    # ── Bikedekho scraper ─────────────────────────────────────────────────────

    def _scrape_bikedekho(self, url_template: str, model: str) -> pd.DataFrame:
        """
        BUG FIX 1: url mutated inside loop
        BUG FIX 2: wrong variable name (final_bikedekho_scrapper)
        BUG FIX 3: 'return' in lambda
        BUG FIX 4: df built inside loop — only last page kept
        BUG FIX 5: returns df
        """
        rows = []
        source = "bikedekho"
        max_pages = self.MAX_PAGES[source]

        for i in range(1, max_pages + 1):
            url = url_template.format(i=i)       
            log.info("  [bikedekho] %s page %d", model, i)
            try:
                r = self._cloud.get(url, timeout=20)
                if r.status_code != 200:
                    log.warning("  HTTP %s — stopping", r.status_code)
                    break
                soup = BeautifulSoup(r.text, "html.parser")
                script = soup.find("script", string=re.compile("__INITIAL_STATE__"))
                if not script:
                    log.warning("  No __INITIAL_STATE__ on page %d — stopping", i)
                    break

                match = re.search(
                    r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\});',
                    script.string, re.DOTALL
                )
                if not match:
                    break

                data    = json.loads(match.group(1))
                reviews = (data.get("userReviews", {})
                               .get("mapItems", {})
                               .get("visibleItems", [{}])[0]
                               .get("subItems", [{}])[0]
                               .get("items", []))

                if not reviews:
                    log.info("  No reviews on page %d — end of results", i)
                    break

                for rev in reviews:
                    rows.append({
                        "user_name":   rev.get("authorName", ""),
                        "review_text": rev.get("fullDescription", ""),
                        "rating":      rev.get("rating"),
                        "posted_date": rev.get("date", ""),
                    })

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                log.warning("  Parse error on page %d: %s", i, e)
                break
            except Exception as e:
                log.warning("  Unexpected error: %s", e)
                break

            time.sleep(2.0)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)                         # BUG FIX: built outside loop
        df["posted_date"] = df["posted_date"].apply(self._bikedekho_date)
        df["source"]     = source
        df["model"]      = model
        df["brand"]      = self.BRAND_MAP[model]
        df["scraped_at"] = self.scraped_at
        df["rating"]     = pd.to_numeric(df["rating"], errors="coerce")
        return df

    # ── Bikewale scraper ──────────────────────────────────────────────────────

    def _scrape_bikewale(self, url_template: str, model: str) -> pd.DataFrame:

        rows = []
        source = "bikewale"
        max_pages = self.MAX_PAGES[source]

        for i in range(1, max_pages + 1):
            url = url_template.format(i=i)       # BUG FIX: no mutation
            log.info("  [bikewale] %s page %d", model, i)
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    log.warning("  HTTP %s — stopping", resp.status_code)
                    break
                soup = BeautifulSoup(resp.text, "html.parser")
            except requests.RequestException as e:
                log.warning("  Request error: %s", e)
                break

            cards = soup.find_all("div", class_="o-cE o-dQ o-c8 o-dk o-bS o-cp o-dz")
            if not cards:
                log.info("  No cards on page %d — end of results", i)
                break

            for card in cards:
                try:
                    u_p       = card.find("div", class_="o-ei o-eS o-j1 o-jK o-f").find_all("p")
                    posted_date = u_p[0].get_text(strip=True) if u_p else ""
                    user_name   = u_p[1].get_text(strip=True) if len(u_p) > 1 else ""

                    # Rating = number of filled star SVGs
                    rating = len(card.find_all("svg", class_="o-aS ZekaAp AoXtgr o-k3"))

                    text_el = card.find("p", class_="o-j1 o-jh o-jJ o-jz")
                    review_text = text_el.get_text(strip=True) if text_el else ""
        

                    rows.append({
                        "user_name":   user_name,
                        "review_text": review_text,
                        "rating":      float(rating),
                        "posted_date": posted_date,
                    })
                except (AttributeError, IndexError):
                    continue

            time.sleep(1.5)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)                         
        df["posted_date"] = df["posted_date"].apply(
            lambda x: self._relative_to_date(x) if x else self.scraped_at
        )
        df["source"]     = source
        df["model"]      = model
        df["brand"]      = self.BRAND_MAP[model]
        df["scraped_at"] = self.scraped_at
        df["rating"]     = pd.to_numeric(df["rating"], errors="coerce")
        return df

    # ── Run scraper ───────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        log.info("=" * 60)
        log.info("EV Scraper — run date: %s", self.scraped_at)
        log.info("=" * 60)

        all_dfs = []

        for model, (url_91, url_bw, url_bd) in self.MODELS.items():
            log.info("── %s ──────────────────────────", model)

            for scrape_fn, url in [
                (self._scrape_91wheels,  url_91),
                (self._scrape_bikewale,  url_bw),
                (self._scrape_bikedekho, url_bd),
            ]:
                try:
                    df = scrape_fn(url, model)
                    if not df.empty:
                        all_dfs.append(df)
                        log.info("  collected %d rows", len(df))
                except Exception as e:
                    log.error("  scraper failed for %s: %s", model, e)

        if not all_dfs:
            log.warning("No data collected in this run.")
            return pd.DataFrame()

        batch = pd.concat(all_dfs, ignore_index=True)
        batch = self._clean_batch(batch)
        return batch

    # ── Cleaning ──────────────────────────────────────────────────────────────

    def _clean_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise and deduplicate within a single scraped batch."""
        df = df.copy()

        # Drop rows with no review text or no user
        df = df[df["review_text"].notna() & (df["review_text"].str.strip() != "")]
        df = df[df["user_name"].notna()  & (df["user_name"].str.strip()  != "")]

        # Normalise text
        df["review_text"] = (df["review_text"]
            .str.replace(r'\s+', ' ', regex=True)
            .str.replace(r'\.{3,}\s*$', '', regex=True)   # strip trailing "....."
            .str.strip())

        df["user_name"] = df["user_name"].str.strip()

        # Clip rating to 1–5
        df["rating"] = df["rating"].clip(1, 5)

        # Stable review ID for deduplication
        df["review_id"] = df.apply(
            lambda r: self._make_review_id(r["source"], r["user_name"], r["review_text"]),
            axis=1
        )

        # Drop duplicates within this batch
        before = len(df)
        df = df.drop_duplicates(subset="review_id")
        log.info("Within-batch dedup: %d → %d rows", before, len(df))

        return df[FINAL_COLS]

    # ── Merge with master ─────────────────────────────────────────────────────

    def merge_with_master(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Appends net-new reviews to the master dataset.
        Reviews already in master (matched by review_id) are skipped.
        """
        if batch.empty:
            log.info("Empty batch — nothing to merge.")
            return pd.DataFrame()

        if MASTER_PATH.exists():
            master = pd.read_csv(MASTER_PATH, dtype=str)
            existing_ids = set(master["review_id"])
            net_new = batch[~batch["review_id"].isin(existing_ids)]
            log.info("Master: %d rows | Batch: %d | Net-new: %d | Duplicates skipped: %d",
                     len(master), len(batch), len(net_new), len(batch) - len(net_new))
            if net_new.empty:
                log.info("No new reviews this run.")
                return net_new
            updated = pd.concat([master, net_new], ignore_index=True)
        else:
            log.info("No master yet — creating from this batch (%d rows)", len(batch))
            net_new = batch
            updated = batch

        updated.to_csv(MASTER_PATH, index=False, encoding="utf-8-sig")
        log.info("Master saved → %s (%d total rows)", MASTER_PATH, len(updated))
        return net_new

    # ── Save batch ────────────────────────────────────────────────────────────

    def save_batch(self, net_new: pd.DataFrame):
        if net_new.empty:
            return
        today = datetime.now().strftime("%Y%m%d")
        path  = BATCH_DIR / f"{today}_batch.csv"
        net_new.to_csv(path, index=False, encoding="utf-8-sig")
        log.info("Batch saved → %s", path)


# ═════════════════════════════════════════════════════════════════════════════
def main():
    scraper = Scraper()
    batch   = scraper.run()
    net_new = scraper.merge_with_master(batch)
    scraper.save_batch(net_new)

    log.info("=" * 60)
    log.info("Done. %d net-new reviews added to master.", len(net_new))
    log.info("=" * 60)


if __name__ == "__main__":
    main()