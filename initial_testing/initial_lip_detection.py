"""
Simple Lip Detection Demo - Sprint 1
Testing lip detection on first 5 GRID videos
"""

import cv2
import dlib
import os
from pathlib import Path


def main():
    print("Simple Lip Detection Test")
    print("=" * 40)
    
    # Initialize dlib
    detector = dlib.get_frontal_face_detector()
    
    # Loading landmark predictor
    predictor = None
    if os.path.exists("../shape_predictor_68_face_landmarks.dat"):
        predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
        print("Lip detection available")
    else:
        print("Face detection only")
    
    grid_path = Path("../data/grid/GRID dataset full/s1")
    videos = sorted(list(grid_path.glob("*.mpg")))[:5]
    
    if not videos:
        print("No videos found in", grid_path)
        return
    
    print(f"Testing {len(videos)} videos")
    
    total_frames = 0
    total_detections = 0
    
    for video in videos:
        print(f"\nVideo: {video.name}")
        
        cap = cv2.VideoCapture(str(video))
        frame_count = 0
        detections = 0
        
        while frame_count < 5 and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # etect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            if faces:
                detections += 1
                face = faces[0]
                
                # Draw face box
                cv2.rectangle(frame, (face.left(), face.top()), 
                             (face.right(), face.bottom()), (0, 255, 0), 2)
                
                # Detect lips if predictor available
                if predictor:
                    landmarks = predictor(gray, face)
                    
                    # Draw lip points (48-67 lip landmarks)
                    for i in range(48, 68):
                        point = landmarks.part(i)
                        cv2.circle(frame, (point.x, point.y), 2, (0, 0, 255), -1)
                
                output_dir = Path("simple_output")
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / f"{video.stem}_frame_{frame_count}.jpg"
                cv2.imwrite(str(output_file), frame)
        
        cap.release()
        
        print(f"Detected: {detections}/{frame_count} frames")
        total_frames += frame_count
        total_detections += detections
    
    print(f"\nSummary:")
    print(f"   Total frames: {total_frames}")
    print(f"   Detections: {total_detections}")
    print(f"   Success rate: {total_detections/total_frames*100:.0f}%")
    print(f"   Output saved to: demo/simple_output/")
    
    if total_detections/total_frames > 0.8:
        print("Success! Detection working well!")
    else:
        print("Needs improvement")


if __name__ == "__main__":
    main()