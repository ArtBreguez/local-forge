"""List open-weight uncensored models on the Hugging Face Hub, to find teacher candidates again.

Stdlib only, no token needed (public API). Examples:

    python3 tools/hf_teachers.py                      # default terms, sorted by downloads
    python3 tools/hf_teachers.py --family qwen3.8 glm-5 --min-downloads 1000
    python3 tools/hf_teachers.py --info huihui-ai/Huihui-Qwen3.8-27B-abliterated
    python3 tools/hf_teachers.py --gguf GLM-5.3-Flash

Prints downloads, likes, last change and id. `--info` adds parameter count, licence and base model.
See docs/TEACHER_CANDIDATES.md for how the list there was chosen.
"""
import argparse
import json
import urllib.parse
import urllib.request

API = "https://huggingface.co/api/models"
TERMS = ["heretic", "abliterated", "uncensored", "derestricted", "heresy"]
# Quantized repacks and image text encoders are copies, not new models.
SKIP = ("gguf", "-mlx", "mlx-community", "exl2", "exl3", "awq", "gptq", "text-encoder", "lora")
UNCENSORED = ("uncens", "ablit", "heretic", "derestr", "heresy")


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def search(term, sort, limit):
    q = urllib.parse.urlencode({"search": term, "sort": sort, "direction": -1, "limit": limit})
    return get(f"{API}?{q}")


def info(model_id):
    d = get(f"{API}/{model_id}")
    total = (d.get("safetensors") or {}).get("total")
    base = [t.split(":", 1)[1] for t in d.get("tags", []) if t.startswith("base_model:") and t.count(":") == 1]
    lic = (d.get("cardData") or {}).get("license", "?")
    size = f"{total / 1e9:.1f}B" if total else "?"
    return f"{model_id}  params={size}  license={lic}  base={','.join(base) or '?'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", nargs="+", default=TERMS)
    ap.add_argument("--family", nargs="*", default=[], help="keep ids holding one of these substrings")
    ap.add_argument("--min-downloads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=60, help="rows to print")
    ap.add_argument("--info", nargs="*", help="print size, licence and base for these ids")
    ap.add_argument("--gguf", help="list uncensored GGUF repacks of a base name")
    a = ap.parse_args()

    if a.info:
        for m in a.info:
            print(info(m))
        return
    if a.gguf:
        for m in search(a.gguf, "downloads", 200):
            mid = m["id"].lower()
            if "gguf" in mid and any(k in mid for k in UNCENSORED):
                print(m["id"])
        return

    seen = {}
    for term in a.terms:
        for sort in ("downloads", "likes", "lastModified"):
            for m in search(term, sort, 100):
                mid = m["id"]
                low = mid.lower()
                if any(s in low for s in SKIP):
                    continue
                if a.family and not any(f.lower() in low for f in a.family):
                    continue
                seen[mid] = (m.get("downloads", 0), m.get("likes", 0), (m.get("lastModified") or m.get("createdAt") or "")[:10])
    rows = sorted(((v[0], v[1], v[2], k) for k, v in seen.items() if v[0] >= a.min_downloads), reverse=True)
    for d, l, when, mid in rows[: a.limit]:
        print(f"{d:>9} {l:>6} {when}  {mid}")


if __name__ == "__main__":
    main()
