"""Phase 1.2 (pilot): verify parsed citations against a ground-truth paper
database.

Reads data/verified_citations/parsed_citations.jsonl (from parse.py) and, for
each citation, searches by title first, falling back to an author+year query
if the title search finds nothing close. Two backends are supported:
  - semantic_scholar (default): api.semanticscholar.org, optionally with an
    API key via the S2_API_KEY env var. The free/unauthenticated tier is
    heavily rate-limited in practice.
  - openalex: api.openalex.org, no key required, much more generous rate
    limits. Set OPENALEX_MAILTO to a contact email to use OpenAlex's
    "polite pool" for faster, more reliable responses.

Labels:
  - citation-level: real / partial / fabricated / unverified_api_error
  - field-level: author / title / venue / year -> correct / incorrect
    (None for all fields when citation-level label is unverified_api_error)

Writes data/verified_citations/verified_citations.jsonl
"""

import argparse
import json
import os
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_HEADERS = {"User-Agent": "citation-hallucination-pilot/0.1 (research)"}
if os.environ.get("S2_API_KEY"):
    S2_HEADERS["x-api-key"] = os.environ["S2_API_KEY"]
S2_FIELDS = "title,authors,venue,year"

OPENALEX_URL = "https://api.openalex.org/works"
OPENALEX_SELECT = "title,publication_year,primary_location,authorships,cited_by_count"

_cache: dict[str, list[dict] | None] = {}


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def author_last_name(author_field: str) -> str:
    # "Kipf and Welling" -> "Kipf"; "Sutton et al." -> "Sutton"
    token = author_field.split(",")[0].split(" and ")[0].split(" et al")[0].strip()
    return token.split()[0] if token else ""


# --- backends -----------------------------------------------------------
# Each search_* function returns a list of normalized candidates:
#   {"title": str, "year": int|None, "venue": str, "authors": [str],
#    "cited_by_count": int}
# or None if the API never returned a usable response (caller must treat
# this as "unknown", not "no match").


def search_semantic_scholar(query: str, retries: int = 6) -> list[dict] | None:
    for attempt in range(retries):
        try:
            resp = requests.get(
                S2_SEARCH_URL,
                params={"query": query, "fields": S2_FIELDS, "limit": 10},
                headers=S2_HEADERS,
                timeout=15,
            )
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code == 200:
            data = resp.json().get("data", []) or []
            return [
                {
                    "title": d.get("title") or "",
                    "year": d.get("year"),
                    "venue": d.get("venue") or "",
                    "authors": [a.get("name", "") for a in (d.get("authors") or [])],
                    "cited_by_count": 0,
                }
                for d in data
            ]
        if resp.status_code == 429:
            wait = min(20 * (attempt + 1), 90)
            print(f"  ...429 rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        return None
    return None


def search_openalex(query: str, retries: int = 4) -> list[dict] | None:
    params = {"filter": f"title.search:{query}", "per-page": 10, "select": OPENALEX_SELECT}
    if os.environ.get("OPENALEX_MAILTO"):
        params["mailto"] = os.environ["OPENALEX_MAILTO"]
    for attempt in range(retries):
        try:
            resp = requests.get(OPENALEX_URL, params=params, timeout=15)
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
            continue
        if resp.status_code == 200:
            results = resp.json().get("results", []) or []
            out = []
            for r in results:
                loc = (r.get("primary_location") or {}).get("source") or {}
                authors = [
                    (a.get("author") or {}).get("display_name", "") for a in (r.get("authorships") or [])
                ]
                out.append(
                    {
                        "title": r.get("title") or "",
                        "year": r.get("publication_year"),
                        "venue": loc.get("display_name") or "",
                        "authors": authors,
                        "cited_by_count": r.get("cited_by_count") or 0,
                    }
                )
            return out
        if resp.status_code == 429:
            wait = 10 * (attempt + 1)
            print(f"  ...429 rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        return None
    return None


BACKENDS = {"semantic_scholar": search_semantic_scholar, "openalex": search_openalex}


def cached_search(backend_fn, query: str) -> list[dict] | None:
    key = f"{backend_fn.__name__}::{query}"
    if key in _cache:
        return _cache[key]
    result = backend_fn(query)
    if result is not None:
        _cache[key] = result
    return result


def best_match(candidates: list[dict], title: str) -> dict | None:
    if not candidates:
        return None
    scored = [(similar(c["title"], title), c) for c in candidates]
    top_score = max(s for s, _ in scored)
    if top_score < 0.6:
        return None
    # break near-ties (e.g. duplicate records for the same paper -- an arXiv
    # preprint entry vs. the peer-reviewed version) preferring a named,
    # non-arXiv venue (what a real citation usually points to), then by
    # citation count.
    tied = [c for s, c in scored if s >= top_score - 0.02]

    def rank(c: dict):
        has_named_venue = bool(c.get("venue")) and "arxiv" not in c["venue"].lower()
        return (has_named_venue, c.get("cited_by_count", 0))

    tied.sort(key=rank, reverse=True)
    return tied[0]


def verify_citation(cit: dict, backend_fn) -> dict:
    candidates = cached_search(backend_fn, cit["title"])
    api_ok = candidates is not None
    match = best_match(candidates, cit["title"]) if candidates else None

    if match is None and api_ok:
        query = f"{author_last_name(cit['author'])} {cit['year']}"
        candidates = cached_search(backend_fn, query)
        api_ok = candidates is not None
        match = best_match(candidates, cit["title"]) if candidates else None

    if not api_ok:
        return {
            **cit,
            "citation_label": "unverified_api_error",
            "field_correct": {"author": None, "title": None, "venue": None, "year": None},
            "matched_paper": None,
        }

    if match is None:
        return {
            **cit,
            "citation_label": "fabricated",
            "field_correct": {"author": False, "title": False, "venue": False, "year": False},
            "matched_paper": None,
        }

    gt_title, gt_year, gt_venue, gt_authors = match["title"], match["year"], match["venue"], match["authors"]
    gt_last_names = {a.split()[-1].lower() for a in gt_authors if a}

    title_ok = similar(gt_title, cit["title"]) >= 0.85
    year_ok = str(gt_year) == str(cit["year"])
    author_ok = author_last_name(cit["author"]).lower() in gt_last_names

    # Many ML papers are indexed with only their arXiv preprint as the
    # primary venue record (no separate conference/journal entry), even
    # when they were also published at a peer-reviewed venue. In that case
    # we can't confirm or deny the model's venue claim, so treat it as
    # unknown rather than penalizing a possibly-correct citation.
    if not gt_venue or "arxiv" in gt_venue.lower():
        venue_ok = None
    else:
        venue_ok = (
            similar(gt_venue, cit["venue"]) >= 0.5
            or gt_venue.lower() in cit["venue"].lower()
            or cit["venue"].lower() in gt_venue.lower()
        )

    field_correct = {"author": author_ok, "title": title_ok, "venue": venue_ok, "year": year_ok}
    if not title_ok:
        label = "fabricated"
    elif any(v is False for v in field_correct.values()):
        label = "partial"
    else:
        label = "real"  # venue may be None (unknown) but nothing is confirmed wrong

    return {
        **cit,
        "citation_label": label,
        "field_correct": field_correct,
        "matched_paper": {"title": gt_title, "year": gt_year, "venue": gt_venue, "authors": gt_authors},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default=str(ROOT / "data/verified_citations/parsed_citations.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/verified_citations/verified_citations.jsonl"))
    ap.add_argument("--backend", choices=list(BACKENDS), default="semantic_scholar")
    ap.add_argument("--sleep", type=float, default=None, help="seconds between API calls (default: backend-specific)")
    args = ap.parse_args()

    sleep_s = args.sleep if args.sleep is not None else (3.5 if args.backend == "semantic_scholar" else 0.3)
    backend_fn = BACKENDS[args.backend]

    citations = [json.loads(l) for l in open(args.parsed)]
    results = []
    for i, cit in enumerate(citations):
        verified = verify_citation(cit, backend_fn)
        results.append(verified)
        print(f"[{i + 1}/{len(citations)}] {cit['author']} ({cit['year']}) -> {verified['citation_label']} {verified['field_correct']}")
        time.sleep(sleep_s)

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    labels = [r["citation_label"] for r in results]
    print(
        f"\nreal={labels.count('real')} partial={labels.count('partial')} "
        f"fabricated={labels.count('fabricated')} unverified_api_error={labels.count('unverified_api_error')}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
