"""
Script to save labelled video frames for phoneme inspection.
- read phoneme timing data from a JSON file
- locate the associated video file from the JSON fields
- for each phoneme pick a timestamp (centre of the interval)
- open the video with cv2.VideoCapture and seek using CAP_PROP_POS_MSEC
- grab the frame, draw a label with phoneme and word using cv2.putText
- save each annotated frame as PNG image in an output directory
"""

import argparse
import json
from pathlib import Path
import cv2


def find_video_path(data):
    if isinstance(data, dict):
        if "video_processed" in data:
            return Path(data["video_processed"])
        if "video_file" in data:
            return Path(data["video_file"])
        # nested
        if "phoneme_extraction" in data and "video_processed" in data:
            return Path(data["video_processed"])
    return None


def collect_phoneme_entries(data):
    """Return list of dicts with keys: phoneme, word, start_time, end_time"""
    entries = []
    if isinstance(data, dict) and "phoneme_extraction" in data:
        data = data["phoneme_extraction"]

    if isinstance(data, dict) and "words" in data:
        for w in data["words"]:
            for p in w.get("phonemes", []):
                entries.append(
                    {
                        "phoneme": p["phoneme"],
                        "word": w.get("word", p.get("word")),
                        "start_time": p["start_time"],
                        "end_time": p["end_time"],
                    }
                )
        return entries

    if (
        isinstance(data, dict)
        and "phonemes" in data
        and isinstance(data["phonemes"], list)
    ):
        for p in data["phonemes"]:
            entries.append(
                {
                    "phoneme": (
                        p.get("label") or p.get("phoneme") or p.get("phone")
                    ),
                    "word": p.get("word") or p.get("lexical") or "",
                    "start_time": float(
                        p.get("start_time", p.get("start", 0.0))
                    ),
                    "end_time": float(
                        p.get("end_time", p.get("end", 0.0))
                    ),
                }
            )
        return entries
    return entries


def extract_frame_opencv(video_path: Path, timestamp: float):
    """
    Return an OpenCV BGR image extracted at timestamp (seconds) or None
    on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def annotate_opencv_image(img, text: str):
    """Annotate BGR image with a dark rectangle and white text at top left"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    margin = 6

    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    rect_h = text_h + margin * 2
    rect_w = text_w + margin * 2

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (rect_w, rect_h), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    text_org = (margin, margin + text_h)
    cv2.putText(
        img,
        text,
        text_org,
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        default="phoneme_extraction_equal_partition.json",
        help="Phoneme JSON file to read",
    )
    parser.add_argument(
        "--outdir",
        default="phoneme_frames",
        help="Output frames directory (under demo/)",
    )
    parser.add_argument(
        "--frame-at",
        choices=["center", "start"],
        default="center",
        help="Where to capture frame in phoneme interval",
    )
    args = parser.parse_args()
    if Path(args.json).is_absolute():
        json_path = Path(args.json)
    else:
        json_path = Path(__file__).parent / args.json
    if not json_path.exists():
        print(f"JSON file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_path = find_video_path(data)
    if video_path is None:
        if isinstance(data, dict):
            pe = data.get("phoneme_extraction", {})
            if "video_file" in pe:
                video_path = Path(pe["video_file"])

    if video_path is None or not video_path.exists():
        print(
            "Video file not found in JSON. Please provide an absolute path"
            " or place the video next to the JSON."
        )
        return

    entries = collect_phoneme_entries(data)
    if not entries:
        print("No phoneme entries found in JSON — nothing to do.")
        return

    out_root = Path(__file__).parent / args.outdir / video_path.stem
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        "Extracting {} frames from {} into {}...".format(
            len(entries), video_path, out_root
        )
    )

    for i, e in enumerate(entries, 1):
        start = float(e.get("start_time", 0.0))
        end = float(e.get("end_time", start))
        if args.frame_at == "center":
            t = (start + end) / 2.0
        else:
            t = start

        safe_ph = str(e.get("phoneme", "UNK"))
        safe_word = str(e.get("word", "")).replace(" ", "_")
        out_img = out_root / f"{i:03d}_{safe_ph}_{safe_word}.png"
        try:
            frame = extract_frame_opencv(video_path, t)
            if frame is None:
                print(f"Failed to read frame #{i} at {t:.3f}s")
                continue

            annotated = annotate_opencv_image(
                frame, "{}  ({})".format(safe_ph, safe_word)
            )
            cv2.imwrite(str(out_img), annotated)
        except Exception as exc:
            print(f"Failed to extract/annotate frame #{i} at {t:.3f}s: {exc}")
    print("Done.")


if __name__ == "__main__":
    main()
