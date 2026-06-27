"""LRCLIB lyrics lookup (no API key). Returns synced (.lrc) and/or plain text;
empty result means the UI should fall back to manual input."""
import requests

_HEADERS = {"User-Agent": "rvc-cover-maker (local tool)"}
_TIMEOUT = 8


def search_lyrics(title: str, artist: str = "") -> dict:
    empty = {"synced": None, "plain": None, "source": None}
    title = (title or "").strip()
    if not title:
        return empty

    # Exact get first (best match when artist is known).
    if artist.strip():
        try:
            r = requests.get(
                "https://lrclib.net/api/get",
                params={"track_name": title, "artist_name": artist.strip()},
                headers=_HEADERS, timeout=_TIMEOUT,
            )
            if r.ok:
                d = r.json()
                return {"synced": d.get("syncedLyrics"), "plain": d.get("plainLyrics"), "source": "lrclib"}
        except requests.RequestException:
            pass

    # Fuzzy search fallback.
    try:
        r = requests.get(
            "https://lrclib.net/api/search",
            params={"q": f"{title} {artist}".strip()},
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        if r.ok and r.json():
            d = r.json()[0]
            return {"synced": d.get("syncedLyrics"), "plain": d.get("plainLyrics"), "source": "lrclib"}
    except (requests.RequestException, ValueError):
        pass

    return empty
