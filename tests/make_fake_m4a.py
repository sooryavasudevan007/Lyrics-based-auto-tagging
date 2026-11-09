"""Create a small fake dataset in the Music4All layout, for testing the pipeline."""
import random, string, sys
from pathlib import Path

out = Path(sys.argv[1]); (out / "lyrics").mkdir(parents=True, exist_ok=True)
rng = random.Random(0)
TAGS = (["rock", "pop", "Hip-Hop", "hip hop", "indie", "alternative", "sad", "happy", "chill",
         "love", "party", "80s", "90s", "female vocalists", "acoustic", "seen live", "metal",
         "electronic", "melancholy", "summer", "country", "jazz", "romantic", "dance", "punk"]
        + [f"rare tag {i}" for i in range(40)])
WORDS = "love night baby heart dance cry tears party fire road home dream rain sun girl boy".split()
artists = [f"Artist {i}" for i in range(400)]
info, tags, genres, lang = [], [], [], []
for n in range(5000):
    sid = "".join(rng.choices(string.ascii_letters + string.digits, k=16))
    a = rng.choice(artists[: rng.choice([50, 400])])          # some prolific artists
    k = rng.randint(0, 8)
    ts = rng.sample(TAGS[:25], k=min(k, 25)) + ([rng.choice(TAGS[25:])] if rng.random() < .2 else [])
    info.append(f"{sid}\t{a}\tSong {n}\tAlbum")
    tags.append(f"{sid}\t{','.join(ts)}")
    genres.append(f"{sid}\t{','.join(t for t in ts if t in ('rock','pop','metal','jazz'))}")
    lang.append(f"{sid}\t{'en' if rng.random() < .85 else 'pt'}")
    if rng.random() < .95:
        n_words = rng.choice([20, 120, 300])
        lines = [" ".join(rng.choices(WORDS, k=8)) for _ in range(n_words // 8)]
        # planted signal: a tag-specific word appears in the lyrics half the time
        lines += [f"word{abs(hash(t.lower().replace('-', ' '))) % 997}" for t in ts if rng.random() < .5]
        (out / "lyrics" / f"{sid}.txt").write_text("[Chorus]\n" + "\n".join(lines))
(out / "id_information.csv").write_text("id\tartist\tsong\talbum_name\n" + "\n".join(info))
(out / "id_tags.csv").write_text("id\ttags\n" + "\n".join(tags))
(out / "id_genres.csv").write_text("id\tgenres\n" + "\n".join(genres))
(out / "id_lang.csv").write_text("id\tlang\n" + "\n".join(lang))
print("fake dataset written to", out)
