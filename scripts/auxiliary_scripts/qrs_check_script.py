import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

MARKUP_DIR = project_root / "data" / "data_with_spikes" / "markings"

QRS_TYPE = 0
SPIKE_TYPE = 1
QRS_AFTER_SPIKE_TYPE = 4

MAX_DISTANCE = 150

def process_file(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments_by_channel = data["Segments"]

    for channel_segments in segments_by_channel:
        channel_segments.sort(key=lambda x: x["StartMark"])

        last_spike_end = None

        for seg in channel_segments:
            seg_type = seg["Type"]

            if seg_type == SPIKE_TYPE:
                last_spike_end = seg["EndMark"]

            elif seg_type == QRS_TYPE:
                if (
                    last_spike_end is not None
                    and 0 <= seg["StartMark"] - last_spike_end <= MAX_DISTANCE
                ):
                    seg["Type"] = QRS_AFTER_SPIKE_TYPE

                last_spike_end = None

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    json_files = sorted(MARKUP_DIR.glob("*.json"))

    for json_file in json_files:
        print(f"Processing: {json_file.name}")
        process_file(json_file)

    print("Done!")


if __name__ == "__main__":
    main()