"""RVC model listing + upload. Models live where the engine expects them:
external/AICoverGen/rvc_models/<name>/{*.pth, *.index}.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RVC_MODELS = ROOT / "external" / "AICoverGen" / "rvc_models"

# Engine support files that live in rvc_models/ but aren't voice models.
_RESERVED = {"hubert_base.pt", "rmvpe.pt"}


def list_models() -> list[dict]:
    out = []
    if not RVC_MODELS.exists():
        return out
    for d in sorted(RVC_MODELS.iterdir()):
        if not d.is_dir() or d.name in _RESERVED:
            continue
        pth = next(iter(d.glob("*.pth")), None)
        if pth:
            out.append({"name": d.name, "has_index": any(d.glob("*.index"))})
    return out


def save_model(name: str, uploads: list[tuple[str, bytes]]) -> dict:
    """uploads: list of (filename, content). Accepts .pth/.index directly, or a
    .zip that contains them. Raises ValueError on bad input."""
    name = name.strip().replace("/", "_")
    if not name:
        raise ValueError("모델 이름이 비어 있습니다.")
    dest = RVC_MODELS / name
    dest.mkdir(parents=True, exist_ok=True)

    has_pth = False
    for fname, content in uploads:
        low = fname.lower()
        if low.endswith(".zip"):
            tmp = dest / fname
            tmp.write_bytes(content)
            with zipfile.ZipFile(tmp) as z:
                for member in z.namelist():
                    ml = member.lower()
                    if ml.endswith((".pth", ".index")) and not member.endswith("/"):
                        target = dest / Path(member).name
                        target.write_bytes(z.read(member))
                        has_pth = has_pth or ml.endswith(".pth")
            tmp.unlink(missing_ok=True)
        elif low.endswith((".pth", ".index")):
            (dest / Path(fname).name).write_bytes(content)
            has_pth = has_pth or low.endswith(".pth")

    if not has_pth:
        raise ValueError(".pth 파일이 없습니다 (.pth 단독 또는 .pth가 든 .zip 업로드).")
    return {"name": name, "has_index": any(dest.glob("*.index"))}
