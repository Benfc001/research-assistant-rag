"""
download_arxiv_data.py

Pulls paper metadata (title, abstract, authors, category, published date)
from arXiv for the ML/NLP subset, using the official arxiv Python package
(a wrapper around arXiv's public API).

Usage:
    python download_arxiv_data.py
"""

import arxiv
import csv
import time

CATEGORIES = ["cs.CL", "cs.LG"]
MAX_RESULTS_PER_CATEGORY = 1500
OUTPUT_FILE = "arxiv_raw.csv"


def fetch_category(category: str, max_results: int):
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)

    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    results = []
    for paper in client.results(search):
        results.append({
            "arxiv_id": paper.get_short_id(),
            "title": paper.title,
            "abstract": paper.summary,
            "authors": "; ".join(a.name for a in paper.authors),
            "primary_category": paper.primary_category,
            "published": paper.published.strftime("%Y-%m-%d"),
            "pdf_url": paper.pdf_url,
        })
    return results


def main():
    all_papers = []
    seen_ids = set()

    for category in CATEGORIES:
        print(f"Fetching up to {MAX_RESULTS_PER_CATEGORY} papers for {category}...")
        papers = fetch_category(category, MAX_RESULTS_PER_CATEGORY)

        new_count = 0
        for p in papers:
            if p["arxiv_id"] not in seen_ids:
                seen_ids.add(p["arxiv_id"])
                all_papers.append(p)
                new_count += 1

        print(f"  -> {new_count} new papers added (total so far: {len(all_papers)})")
        time.sleep(2)

    fieldnames = ["arxiv_id", "title", "abstract", "authors",
                  "primary_category", "published", "pdf_url"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_papers)

    print(f"\nDone. Wrote {len(all_papers)} papers to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
