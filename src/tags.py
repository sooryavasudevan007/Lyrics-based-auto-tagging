"""Rule-based tag normalisation and the draft tag-category mapping.

Both use only the tag strings (never model results), as the README requires.
"""
from collections import Counter
import re

_SPACE = re.compile(r"[\s_\-/]+")
_KEY = re.compile(r"[^a-z0-9]")


def normalise_one(tag: str) -> str:
    t = tag.lower().strip().replace("&", " and ")
    t = _SPACE.sub(" ", t).strip()
    return t


def build_merge_map(tag_lists) -> dict[str, str]:
    """Map every normalised spelling to one canonical form.

    Spellings that differ only in spaces/punctuation (hip hop, hip-hop, hiphop)
    share a key; the most frequent spelling becomes canonical.
    """
    counts = Counter(normalise_one(t) for tags in tag_lists for t in tags)
    by_key: dict[str, list[str]] = {}
    for t in counts:
        by_key.setdefault(_KEY.sub("", t), []).append(t)
    merge = {}
    for variants in by_key.values():
        canon = max(variants, key=lambda v: (counts[v], -len(v), v))
        for v in variants:
            merge[v] = canon
    return merge


def load_synonyms(path):
    """variant -> canonical, for different words with the same meaning
    (e.g. melancholic -> melancholy). Applied after spelling merges."""
    import csv
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {normalise_one(r["variant"]): normalise_one(r["canonical"]) for r in csv.DictReader(f)}


def apply_normalisation(tag_lists, merge: dict[str, str], synonyms=None):
    synonyms = synonyms or {}
    out = []
    for tags in tag_lists:
        seen = []
        for t in tags:
            c = merge.get(normalise_one(t))
            c = synonyms.get(c, c)
            if c and c not in seen:
                seen.append(c)
        out.append(seen)
    return out


# ---------------------------------------------------------------- categories
# Keyword lexicon for the semi-automatic mapping (Soorya, Obj. 1 PI5).
# Suggestions are corrected by a person in splits/category_overrides.csv;
# both steps use the tag text only, never model results.
CATEGORIES = ["genre", "mood/emotion", "theme", "era", "instrumentation/production", "other"]

LEXICON = {
    "era": [r"^\d0s$", r"^(19|20)\d0s$", r"^(19|20)\d\d$", r"^oldies$"],
    "instrumentation/production": [
        r"guitar", r"piano", r"^acoustic$", r"instrumental", r"vocal", r"female fronted",
        r"synthesizer", r"\bbass\b", r"drum", r"strings", r"orchestra", r"saxophone",
        r"violin", r"^live$", r"^cover", r"remix", r"lo ?fi", r"a cappella", r"^male", r"^female"],
    "mood/emotion": [
        r"\bsad\b", r"happy", r"melanchol", r"^chill(out)?$", r"relax", r"calm", r"mellow",
        r"energetic", r"upbeat", r"^dark$", r"angry", r"aggressive", r"romantic", r"sexy",
        r"dreamy", r"atmospheric", r"uplifting", r"feel ?good", r"depress", r"beautiful",
        r"emotional", r"nostalg", r"^epic$", r"^fun$", r"heartbreak", r"lonely", r"peaceful",
        r"catchy", r"haunting", r"sweet", r"cute", r"soothing", r"moody", r"bittersweet",
        r"hopeful", r"intense", r"powerful", r"\bcool\b", r"\bsoft\b$"],
    "theme": [
        r"^love$", r"love song", r"\bparty\b", r"summer", r"christmas", r"\bdrugs?\b",
        r"political", r"protest", r"religio", r"christian", r"break ?up", r"friendship",
        r"^death$", r"\bnight\b", r"driving", r"workout", r"sleep", r"study", r"travel",
        r"\bwar\b", r"\bsex\b", r"lyrics", r"story", r"heart ?break song", r"\bgod\b"],
    "genre": [
        r"rock", r"pop", r"metal", r"punk", r"jazz", r"blues", r"soul", r"r ?and ?b", r"rnb",
        r"hip ?hop", r"trip ?hop", r"rap", r"country", r"folk", r"indie", r"alternative",
        r"electro", r"house", r"techno", r"trance", r"dubstep", r"edm", r"reggae", r"funk",
        r"disco", r"grunge", r"emo\b", r"screamo", r"hardcore", r"ambient", r"classical",
        r"latin", r"britpop", r"new wave", r"shoegaze", r"trap", r"grime", r"gothic",
        r"psychedel", r"progressive", r"garage", r"ska", r"soundtrack", r"americana",
        r"swing", r"industrial", r"experimental", r"singer songwriter", r"^dance$",
        r"synth", r"ballad", r"gospel", r"chillout", r"downtempo", r"lounge", r"bossa",
        r"world", r"noise", r"post ", r"wave$"],
}
# First match wins. Genre comes before theme so that "death metal" is a genre,
# and mood before genre so that "chillout" is a mood (see overrides for exceptions).
_ORDER = ["era", "instrumentation/production", "mood/emotion", "genre", "theme"]
_COMPILED = {c: [re.compile(p) for p in LEXICON[c]] for c in _ORDER}

# Tags that describe the listener or the source, not the song
_OTHER = re.compile(
    r"seen live|favou?rite|\bmy\b|awesome|spotify|\bbest\b|amazing|love at first|"
    r"playlist|check out|under \d+|^fip$|"
    r"^(american|british|canadian|australian|swedish|german|french|english|irish|"
    r"scottish|norwegian|finnish|danish|dutch|japanese|korean|brazilian|spanish|"
    r"italian|uk|usa|us)$")


def suggest_category(tag: str) -> str:
    if _OTHER.search(tag):
        return "other"
    for cat in _ORDER:
        if any(p.search(tag) for p in _COMPILED[cat]):
            return cat
    return "other"


def load_overrides(path):
    """tag -> (category, reason) from the manual review file, if present."""
    import csv
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bad = [r for r in rows if r["category"] not in CATEGORIES]
    if bad:
        raise ValueError(f"Unknown categories in {path.name}: {bad}")
    return {r["tag"]: (r["category"], r.get("reason", "")) for r in rows}


def categorise(tag, overrides):
    if tag in overrides:
        return overrides[tag][0], "manual"
    return suggest_category(tag), "lexicon"
