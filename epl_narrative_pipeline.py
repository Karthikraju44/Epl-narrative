"""
Team Blaugrana - Web crawling + text analyser + Clustering
UNIQUE ANGLE: "Narrative Fingerprints" of Premier League clubs

Most teams will cluster articles by TOPIC (transfers, injuries, results).
We cluster by NARRATIVE FRAME (crisis, title race, redemption, injury blow,
hype...) and then build a fingerprint per club: how does the media talk
about each club? We add sentiment so each frame has a tone score.

Pipeline:
  1. Crawl BBC Sport EPL hub pages -> article links
  2. Download + parse articles (title, date, body)
  3. Clean text, tag the primary club of each article
  4. TF-IDF (unigrams + bigrams) -> KMeans (k picked by silhouette)
  5. Sentiment (VADER) per article
  6. Outputs: CSVs + 3 plots (PCA scatter, club x narrative heatmap, club sentiment)

Install:
  pip install requests beautifulsoup4 pandas scikit-learn matplotlib vaderSentiment

Run:
  python epl_narrative_pipeline.py --max-articles 150
"""

import argparse
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE = "https://www.bbc.com"
HUB_PAGES = [
    "/sport/football/premier-league",
    "/sport/football/premier-league/scores-fixtures",
    "/sport/football",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (student project - VIT Amaravati; contact: your-email)"}
OUT = Path("output")
OUT.mkdir(exist_ok=True)

# Club name -> aliases used in articles
CLUBS = {
    "Arsenal": ["arsenal", "gunners"],
    "Aston Villa": ["aston villa", "villa"],
    "Bournemouth": ["bournemouth", "cherries"],
    "Brentford": ["brentford", "bees"],
    "Brighton": ["brighton", "seagulls"],
    "Burnley": ["burnley"],
    "Chelsea": ["chelsea", "blues"],
    "Crystal Palace": ["crystal palace", "palace"],
    "Everton": ["everton", "toffees"],
    "Fulham": ["fulham"],
    "Leeds": ["leeds"],
    "Liverpool": ["liverpool", "reds"],
    "Man City": ["manchester city", "man city", "city"],
    "Man Utd": ["manchester united", "man utd", "united"],
    "Newcastle": ["newcastle", "magpies"],
    "Nottm Forest": ["nottingham forest", "forest"],
    "Sunderland": ["sunderland"],
    "Spurs": ["tottenham", "spurs"],
    "West Ham": ["west ham", "hammers"],
    "Wolves": ["wolves", "wolverhampton"],
}

# Optional: rename clusters after you inspect top terms (cluster_id -> label)
NARRATIVE_LABELS = {}


# ------------------------------------------------------------------ crawling
def get(url, retries=2):
    for _ in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            pass
        time.sleep(2)
    return None


def collect_links(max_links):
    links = []
    for hub in HUB_PAGES:
        html = get(BASE + hub)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].split("?")[0]
            if re.search(r"/sport/football/articles/[a-z0-9]+$", href):
                full = href if href.startswith("http") else BASE + href
                if full not in links:
                    links.append(full)
        time.sleep(1)
    print(f"Found {len(links)} candidate article links")
    return links[:max_links]


def parse_article(url):
    html = get(url)
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    blocks = soup.select('[data-component="text-block"]')
    paras = [b.get_text(" ", strip=True) for b in blocks]
    if not paras:  # fallback if BBC changes markup
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    body = " ".join(paras)

    t = soup.find("time")
    date = t.get("datetime", "") if t else ""
    if len(body.split()) < 120:  # skip live pages / video pages
        return None
    return {"url": url, "title": title, "date": date, "body": body}


def crawl(max_articles):
    rows = []
    for i, link in enumerate(collect_links(max_articles * 2), 1):
        art = parse_article(link)
        if art:
            rows.append(art)
            print(f"[{len(rows)}] {art['title'][:70]}")
        if len(rows) >= max_articles:
            break
        time.sleep(1.2)  # be polite
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ cleaning
def clean(text):
    text = re.sub(r"http\S+", " ", text.lower())
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def primary_club(title, body):
    """Club with the most mentions; title mentions count triple."""
    t, b = title.lower(), body.lower()
    scores = Counter()
    for club, aliases in CLUBS.items():
        for al in aliases:
            pat = r"\b" + re.escape(al) + r"\b"
            scores[club] += 3 * len(re.findall(pat, t)) + len(re.findall(pat, b))
    club, score = scores.most_common(1)[0]
    return club if score >= 3 else "General/Multiple"


# ------------------------------------------------------------------ modelling
def build_stopwords():
    # drop club names so clusters reflect NARRATIVE, not which club is mentioned
    club_words = {w for al in CLUBS.values() for a in al for w in a.split()}
    extra = {"said", "says", "bbc", "sport", "premier", "league", "game", "games",
             "season", "week", "told", "like", "just", "also", "year", "years",
             "mr", "football", "club", "team", "manager", "boss"}
    return list(ENGLISH_STOP_WORDS | club_words | extra)


def best_k(X, k_range=range(3, 9)):
    best, best_score = None, -1
    for k in k_range:
        if k >= X.shape[0]:
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        s = silhouette_score(X, labels)
        print(f"k={k}: silhouette={s:.3f}")
        if s > best_score:
            best, best_score = k, s
    return best


def top_terms(km, vec, n=10):
    terms = np.array(vec.get_feature_names_out())
    order = km.cluster_centers_.argsort()[:, ::-1]
    return {i: ", ".join(terms[order[i, :n]]) for i in range(km.n_clusters)}


# ------------------------------------------------------------------ main
def main(max_articles, k_override):
    df = crawl(max_articles)
    if df.empty:
        raise SystemExit("No articles scraped - check selectors / network.")
    df.to_csv(OUT / "raw_articles.csv", index=False)

    df["clean"] = (df["title"] + " " + df["body"]).apply(clean)
    df["club"] = [primary_club(t, b) for t, b in zip(df["title"], df["body"])]

    vec = TfidfVectorizer(stop_words=build_stopwords(), ngram_range=(1, 2),
                          min_df=2, max_df=0.6, max_features=4000)
    X = vec.fit_transform(df["clean"])

    # reduce noise before KMeans (standard for sparse text)
    n_comp = min(50, X.shape[1] - 1, X.shape[0] - 1)
    Xr = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(X)
    Xr = Xr / (np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-9)

    k = k_override or best_k(Xr)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)  # terms from sparse space
    df["cluster"] = km.labels_
    terms = top_terms(km, vec)
    df["narrative"] = df["cluster"].map(lambda c: NARRATIVE_LABELS.get(c, f"C{c}: " + ", ".join(terms[c].split(", ")[:3])))

    sia = SentimentIntensityAnalyzer()
    df["sentiment"] = df["body"].apply(lambda t: sia.polarity_scores(t)["compound"])

    # ---- outputs
    df.drop(columns=["body", "clean"]).to_csv(OUT / "clustered_articles.csv", index=False)
    pd.Series(terms, name="top_terms").to_csv(OUT / "cluster_top_terms.csv")
    print("\nTop terms per cluster (use these to name your narratives):")
    for c, t in terms.items():
        print(f"  C{c} ({(df.cluster == c).sum()} articles): {t}")

    # Plot 1: PCA scatter
    pts = PCA(n_components=2, random_state=42).fit_transform(Xr)
    plt.figure(figsize=(9, 6))
    for c in range(k):
        m = df["cluster"] == c
        plt.scatter(pts[m, 0], pts[m, 1], label=df.loc[m, "narrative"].iloc[0], alpha=0.7)
    plt.legend(fontsize=7, loc="best")
    plt.title("Article clusters by narrative frame (PCA)")
    plt.tight_layout()
    plt.savefig(OUT / "plot_clusters.png", dpi=150)
    plt.close()

    # Plot 2: club x narrative fingerprint heatmap (club rows normalised to 100%)
    clubs_df = df[df["club"] != "General/Multiple"]
    if len(clubs_df) > 0:
        ct = pd.crosstab(clubs_df["club"], clubs_df["narrative"], normalize="index") * 100
        ct = ct[ct.index.isin(clubs_df["club"].value_counts()[lambda s: s >= 2].index)]
        if not ct.empty:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(ct))))
            im = ax.imshow(ct.values, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(ct.shape[1]))
            ax.set_xticklabels(ct.columns, rotation=40, ha="right", fontsize=7)
            ax.set_yticks(range(ct.shape[0]))
            ax.set_yticklabels(ct.index, fontsize=8)
            plt.colorbar(im, label="% of club's articles")
            ax.set_title("Narrative fingerprint per club")
            plt.tight_layout()
            plt.savefig(OUT / "plot_club_fingerprint.png", dpi=150)
            plt.close()
            ct.round(1).to_csv(OUT / "club_fingerprint.csv")

        # Plot 3: average sentiment per club
        s = clubs_df.groupby("club")["sentiment"].agg(["mean", "count"])
        s = s[s["count"] >= 2].sort_values("mean")
        if not s.empty:
            plt.figure(figsize=(8, max(4, 0.35 * len(s))))
            plt.barh(s.index, s["mean"], color=np.where(s["mean"] >= 0, "tab:green", "tab:red"))
            plt.axvline(0, color="k", lw=0.8)
            plt.title("Average media sentiment per club")
            plt.tight_layout()
            plt.savefig(OUT / "plot_club_sentiment.png", dpi=150)
            plt.close()

    print(f"\nDone. Results in ./{OUT}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-articles", type=int, default=150)
    ap.add_argument("--k", type=int, default=None, help="force number of clusters")
    args = ap.parse_args()
    main(args.max_articles, args.k)
