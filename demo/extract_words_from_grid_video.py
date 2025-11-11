"""
Sprint 2 – Word Extraction Demo
- Parses GRID dataset .align files to extract word-level alignments.
- Displays start/end times, durations, and reconstructs spoken sentences.
- Saves structured results as JSON for further preprocessing stages.
"""

import sys
import os
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent))


def find_grid_videos_and_alignments():
    """Find matching video and alignment files in the GRID dataset."""
    grid_data_path = Path(__file__).parent.parent / "data" / "grid" / "GRID dataset full"
    videos_path = grid_data_path / "s1"
    alignments_path = grid_data_path / "alignments" / "s1"
    
    video_files = [f for f in os.listdir(videos_path) if f.endswith('.mpg')]
    
    matching_pairs = []
    for video_file in video_files:
        base_name = os.path.splitext(video_file)[0]
        alignment_file = base_name + ".align"
        
        video_path = videos_path / video_file
        alignment_path = alignments_path / alignment_file
        
        if alignment_path.exists():
            matching_pairs.append({
                'video_path': video_path,
                'alignment_path': alignment_path,
                'base_name': base_name
            })
    
    return matching_pairs[:10]  # Return first 10 for demo


def parse_grid_alignment_file(alignment_file: Path) -> list:
    """Parse a GRID .align file to extract word-level alignments."""
    word_alignments = []
    
    with open(alignment_file, 'r') as f:
        content = f.read().strip()
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) >= 3:
            start_frame = int(parts[0])
            end_frame = int(parts[1])
            word = parts[2]
            
            # Convert frames to seconds (25000 frames = 1 second)
            start_time = start_frame / 25000.0
            end_time = end_frame / 25000.0
            duration = end_time - start_time
            
            word_alignments.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'word': word
            })
    
    return word_alignments


def display_word_alignments(word_alignments: list, video_file: Path):
    """Display word alignments in a formatted table."""
    if word_alignments:
        total_duration = word_alignments[-1]['end_time'] - word_alignments[0]['start_time']
        print(f"Processed video: {video_file.name}")
        print(f"Total video duration: {total_duration:.3f} seconds")
    
    print(f"All words extarcted:")
    print(f"{'#':<3} | {'Word':<12} | {'Type':<8} | {'Start':<8} | {'End':<8} | {'Duration':<8}")
    print("-" * 70)
    
    sentence_words = []
    
    for i, alignment in enumerate(word_alignments, 1):
        word = alignment['word']
        word_type = "SILENCE" if word.lower() == 'sil' else "SPEECH"
        
        print(f"{i:<3} | {word:<12} | {word_type:<8} | {alignment['start_time']:<8.3f} | {alignment['end_time']:<8.3f} | {alignment['duration']:<8.3f}")
        
        if word.lower() != 'sil':
            sentence_words.append(word)
    
    if sentence_words:
        print(f"\nSENTENCE: {' '.join(sentence_words)}")
        print(f"SPEECH WORDS: {len(sentence_words)}")


def main():
    """Main demo function."""
    video_alignment_pairs = find_grid_videos_and_alignments()
    selected_pair = video_alignment_pairs[1]
    video_file = selected_pair['video_path']
    alignment_file = selected_pair['alignment_path']
    word_alignments = parse_grid_alignment_file(alignment_file)
    display_word_alignments(word_alignments, video_file)
    speech_words = [w for w in word_alignments if w['word'].lower() != 'sil']
    
    demo_data = {
        'video_file': str(video_file),
        'alignment_file': str(alignment_file),
        'word_alignments': word_alignments,
        'sentence': ' '.join([w['word'] for w in speech_words])
    }
    
    output_file = Path(__file__).parent / "demo_results.json"
    with open(output_file, 'w') as f:
        json.dump(demo_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file.name}")


if __name__ == "__main__":
    main()