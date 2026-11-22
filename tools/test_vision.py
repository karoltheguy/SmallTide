"""test_vision.py: image input end to end against a running SmallTide server.

Draws test images with Pillow (a word and a number in large type on a coloured background, a
shape), sends them as data: URLs and checks that the answers name what is in them, that a
repeated request reuses the cached prefix, and that a different image of the same size does not.

    python -I tools/test_vision.py [--base http://127.0.0.1:8012]

Needs Pillow.
"""
import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request

from PIL import Image, ImageDraw, ImageFont


def font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw(word, number, bg, shape, size=(640, 400)):
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    d.text((40, 40), word, fill="white", font=font(96))
    d.text((40, 170), str(number), fill="yellow", font=font(120))
    box = (440, 180, 600, 340)
    if shape == "circle":
        d.ellipse(box, fill="red", outline="white", width=4)
    else:
        d.rectangle(box, fill="green", outline="white", width=4)
    return im


def data_url(im, fmt):
    b = io.BytesIO()
    im.save(b, fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(b.getvalue()).decode()}"


def ask(base, content, max_tokens=300):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": "none"}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as e:
        return None, e.read().decode(errors="replace"), {}
    m = r["choices"][0]["message"]
    return (m.get("content") or "") + (m.get("reasoning_content") or ""), None, r.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8012")
    a = ap.parse_args()
    q = "Read this image. Reply in one line: the word, the number, the background colour, and the shape with its colour."
    cases = [
        ("PNG", draw("TIDE", 4721, "navy", "circle"), ["tide", "4721", "circle"]),
        ("JPEG", draw("MOON", 9053, "purple", "square"), ["moon", "9053"]),
    ]
    ok = True
    for fmt, im, want in cases:
        content = [{"type": "text", "text": q}, {"type": "image_url", "image_url": {"url": data_url(im, fmt)}}]
        text, err, usage = ask(a.base, content)
        if err:
            print(f"{fmt}: HTTP error {err[:300]}")
            ok = False
            continue
        hit = [w for w in want if w in text.lower()]
        good = len(hit) == len(want)
        ok &= good
        print(f"{fmt}: {'PASS' if good else 'FAIL'} prompt_tokens={usage.get('prompt_tokens')} found {hit} of {want}\n    answer: {text.strip()[:200]!r}")
        # the same request again: the prefix cache holds it
        text2, _, usage2 = ask(a.base, content)
        cached = (usage2.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        # (the text may differ in wording: which experts are resident changes between requests)
        same = all(w in (text2 or "").lower() for w in want)
        reused = cached >= usage2.get("prompt_tokens", 1 << 30) - 2
        ok &= same and reused
        print(f"    repeat: {'PASS' if same and reused else 'FAIL'} cached {cached} of {usage2.get('prompt_tokens')} tokens, still names {want}: {same}")
    # a different image of the same size after the same text: nothing of the first image's is reused
    c1 = [{"type": "text", "text": q}, {"type": "image_url", "image_url": {"url": data_url(draw("LAKE", 1385, "black", "circle"), "PNG")}}]
    text, err, usage = ask(a.base, c1)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    good = err is None and "lake" in text.lower() and "1385" in text and cached < 60
    ok &= good
    print(f"other image, same size: {'PASS' if good else 'FAIL'} cached {cached} of {usage.get('prompt_tokens')}, answer {text.strip()[:120] if text else err!r}")
    # two images in one message
    c2 = [{"type": "text", "text": "Two images follow. Reply in one line with the word in the first and the word in the second."},
          {"type": "image_url", "image_url": {"url": data_url(draw("FROG", 11, "teal", "circle", (320, 200)), "PNG")}},
          {"type": "image_url", "image_url": {"url": data_url(draw("WOLF", 22, "maroon", "square", (320, 200)), "PNG")}}]
    text, err, usage = ask(a.base, c2)
    good = err is None and "frog" in text.lower() and "wolf" in text.lower() and text.lower().index("frog") < text.lower().index("wolf")
    ok &= good
    print(f"two images: {'PASS' if good else 'FAIL'} prompt_tokens={usage.get('prompt_tokens')}, answer {text.strip()[:160] if text else err!r}")
    # a bad image is a 400, not a crash
    text, err, _ = ask(a.base, [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}])
    good = err is not None and "image" in err
    ok &= good
    print(f"bad image: {'PASS' if good else 'FAIL'} {err[:120] if err else text!r}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
