"""
Script để tạo dataset test cases từ JVS Corpus cho đánh giá speaker diarization và ASR.
Dataset JVS: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus

Usage:
    python create_dataset.py --jvs_root ../dataset/jvs_ver1 --output dataset_400_testcases.csv
"""

import os
import csv
import argparse
import random
from pathlib import Path

# Random seed for reproducibility
random.seed(42)

# Categories in JVS dataset
JVS_CATEGORIES = ["parallel100", "nonpara30", "whisper10", "falset10"]


def load_transcripts(txt_path):
    """
    Đọc transcripts_utf8.txt → trả về dict: file_name → transcript
    """
    mapping = {}
    if not os.path.exists(txt_path):
        return mapping

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            # Format: filename:transcript
            fname, text = line.split(":", 1)
            mapping[fname] = text.strip()
    
    return mapping


def pick_one_random_file(speaker_dir, category):
    """
    Chọn 1 file WAV + transcript bất kỳ trong thư mục category.
    Returns dict với thông tin file hoặc None nếu không tìm thấy.
    """
    cat_dir = os.path.join(speaker_dir, category)
    
    transcript_path = os.path.join(cat_dir, "transcripts_utf8.txt")
    wav_dir = os.path.join(cat_dir, "wav24kHz16bit")
    
    if not os.path.exists(transcript_path) or not os.path.exists(wav_dir):
        return None
    
    # Load transcripts
    transcripts = load_transcripts(transcript_path)
    if not transcripts:
        return None
    
    # Lọc các file .wav tồn tại
    wav_files = [
        f for f in os.listdir(wav_dir)
        if f.endswith(".wav") and f.replace(".wav", "") in transcripts
    ]
    
    if not wav_files:
        return None
    
    # Chọn ngẫu nhiên 1 file
    chosen = random.choice(wav_files)
    file_id = chosen.replace(".wav", "")
    
    return {
        "speaker": os.path.basename(speaker_dir),
        "category": category,
        "file_name": file_id,
        "wav_path": os.path.join(wav_dir, chosen),
        "transcript": transcripts[file_id]
    }


def build_jvs_dataset(jvs_root, samples_per_category=1):
    """
    Tạo dataset từ JVS corpus.
    
    Args:
        jvs_root: Đường dẫn tới thư mục gốc của JVS (chứa jvs001, jvs002, ...)
        samples_per_category: Số samples mỗi category cho mỗi speaker
    
    Returns:
        List of dict với thông tin audio files
    """
    dataset = []
    
    jvs_path = Path(jvs_root)
    if not jvs_path.exists():
        print(f"Error: JVS directory not found: {jvs_root}")
        return dataset
    
    # Lấy danh sách speakers
    speakers = sorted([
        d for d in jvs_path.iterdir()
        if d.is_dir() and d.name.startswith('jvs')
    ])
    
    print(f"Found {len(speakers)} speakers in JVS dataset")
    
    # Duyệt qua từng speaker và category
    for speaker_dir in speakers:
        for category in JVS_CATEGORIES:
            for _ in range(samples_per_category):
                item = pick_one_random_file(str(speaker_dir), category)
                if item:
                    dataset.append(item)
                else:
                    print(f"[WARN] Missing data: {speaker_dir.name}/{category}")
    
    return dataset


def save_csv(dataset, output_path):
    """Lưu dataset thành CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["speaker", "category", "file_name", "wav_path", "transcript"]
        )
        writer.writeheader()
        writer.writerows(dataset)
    
    print(f"✓ Saved dataset to: {output_path}")


def print_statistics(dataset):
    """In thống kê về dataset."""
    if not dataset:
        return
    
    speakers = set(item['speaker'] for item in dataset)
    categories = {}
    for item in dataset:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples:      {len(dataset)}")
    print(f"Total speakers:     {len(speakers)}")
    print(f"Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat:15s}: {count:4d} samples")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Create evaluation dataset from JVS Corpus'
    )
    parser.add_argument(
        '--jvs_root',
        type=str,
        default='../dataset/jvs_ver1',
        help='Path to JVS root directory (containing jvs001, jvs002, etc.)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dataset_400_testcases.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--samples_per_category',
        type=int,
        default=1,
        help='Number of samples per category per speaker (default: 1)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("JVS DATASET CREATOR")
    print(f"{'='*60}")
    print(f"JVS Root:           {args.jvs_root}")
    print(f"Output:             {args.output}")
    print(f"Samples/Category:   {args.samples_per_category}")
    print(f"{'='*60}\n")
    
    # Build dataset
    print("Building dataset from JVS corpus...")
    dataset = build_jvs_dataset(args.jvs_root, args.samples_per_category)
    
    if not dataset:
        print("Error: No data collected!")
        return
    
    print(f"✓ Collected {len(dataset)} samples")
    
    # Print statistics
    print_statistics(dataset)
    
    # Save CSV
    output_path = Path(__file__).parent / args.output
    save_csv(dataset, output_path)
    
    print(f"\n✓ Dataset creation completed successfully!")
    print(f"  Use this file for ASR evaluation:")
    print(f"  python eval_asr.py --dataset {args.output} --model whisper\n")


if __name__ == "__main__":
    main()
