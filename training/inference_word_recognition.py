"""
Word Recognition Inference Script
This script loads a trained 3D CNN model and performs inference on
new video samples to predict the spoken word from lip movements.
"""

import argparse
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import numpy as np
import torch
import cv2


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
from training.train_word_recognition_3d_cnn import (  # noqa: E402
    ThreeDimensionalCNN)


def load_trained_model(checkpoint_path: Path, device: torch.device):
    """Load a trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    word_to_index = checkpoint['word_to_index']
    index_to_word = checkpoint['index_to_word']
    index_to_word = {int(k): v for k, v in index_to_word.items()}
    # create model
    num_classes = len(word_to_index)
    model_config = checkpoint.get("config", {})
    input_channels = int(model_config.get("input_channels", 3))
    model = ThreeDimensionalCNN(
        input_channels=input_channels,
        number_of_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    print(f"Number of classes: {num_classes}")
    validation_accuracy = checkpoint.get(
        "validation_accuracy",
        checkpoint.get("val_accuracy", 0.0))
    print(f"Validation accuracy: {validation_accuracy:.2f}%")
    return model, word_to_index, index_to_word, model_config


def resolve_inference_device() -> torch.device:
    """Use CUDA if available, otherwise stay on CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device
    print("Using CPU for inference")
    return torch.device("cpu")


def load_video_frames_from_file(video_path: Path) -> np.ndarray:
    """Load frames from .npy or standard video file"""
    if video_path.suffix.lower() == ".npy":
        return np.load(video_path, allow_pickle=False)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames found in video: {video_path}")
    return np.asarray(frames)


def record_webcam_clip(
    output_path: Path,
    camera_index: int,
    seconds: float,
) -> Path:
    """Record a short webcam clip for demo inference"""
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0
    total_frames = max(1, int(seconds * fps))
    recorded_frames = []
    print(
        f"Recording webcam clip for {seconds:.1f}seconds "
        f"(~{total_frames} frames)")

    for _ in range(total_frames):
        ok, frame = capture.read()
        if not ok:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        recorded_frames.append(rgb_frame)
    capture.release()

    if not recorded_frames:
        raise RuntimeError("No frames captured from webcam")
    video_frames = np.asarray(recorded_frames)
    np.save(output_path, video_frames)
    return output_path


def preprocess_video_frames(
    video_frames: np.ndarray,
    target_num_frames: int = 40,
    target_height: int = 48,
    target_width: int = 48,
) -> torch.Tensor:
    """Preprocess video frames for model input"""
    current_num_frames = video_frames.shape[0]
    # standardize number of frames
    if current_num_frames > target_num_frames:
        frame_indices = np.linspace(
            0,
            current_num_frames - 1,
            target_num_frames,
            dtype=int)
        video_frames = video_frames[frame_indices]
    elif current_num_frames < target_num_frames:
        frames_needed = target_num_frames - current_num_frames
        padding_frames = np.repeat(video_frames[-1:], frames_needed, axis=0)
        video_frames = np.concatenate([video_frames, padding_frames], axis=0)
    resized_frames = []
    for frame in video_frames:
        resized_frame = cv2.resize(frame, (target_width, target_height))
        resized_frames.append(resized_frame)
    processed_frames = np.array(resized_frames)
    processed_frames = processed_frames.astype(np.float32) / 255.0
    frames_tensor = torch.from_numpy(processed_frames).float()
    frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]
    frames_tensor = frames_tensor.unsqueeze(0)
    return frames_tensor


def predict_word(
    model: torch.nn.Module,
    video_frames: torch.Tensor,
    index_to_word: dict,
    device: torch.device,
    top_k: int = 5,
) -> list:
    """Predict word from video frames"""
    video_frames = video_frames.to(device)
    with torch.no_grad():
        logits = model(video_frames)
        probabilities = torch.softmax(logits, dim=1)
    top_probs, top_indices = torch.topk(probabilities[0], top_k)
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        word = index_to_word[idx.item()]
        confidence = prob.item() * 100
        predictions.append((word, confidence))
    return predictions


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(
        description="Word Recognition Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            PROJECT_ROOT
            / "training"
            / "checkpoints_words"
            / "best_model_words.pth"),
        help="Path to model checkpoint")
    parser.add_argument(
        "--video_path",
        type=str,
        help="Path to .npy or video file")
    parser.add_argument(
        "--record_webcam",
        action="store_true",
        help="Record a short clip from webcam for demo inference")
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Camera index for webcam capture")
    parser.add_argument(
        "--record_seconds",
        type=float,
        default=2.0,
        help="Duration of webcam recording in seconds")
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top predictions to show")
    args = parser.parse_args()
    if not args.video_path and not args.record_webcam:
        parser.error("Provide --video_path or use --record_webcam")
    device = resolve_inference_device()
    print(f"Using device: {device}\n")
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(
            "Please train a model first using "
            "train_word_recognition_3d_cnn.py")
        return
    model, word_to_index, index_to_word, model_config = load_trained_model(
        checkpoint_path,
        device)
    # load video
    if args.record_webcam:
        with NamedTemporaryFile(suffix=".npy", delete=False) as temp_file:
            video_path = Path(temp_file.name)
        video_path = record_webcam_clip(
            output_path=video_path,
            camera_index=args.camera_index,
            seconds=args.record_seconds)
    else:
        video_path = Path(args.video_path)
        if not video_path.exists():
            print(f"Error: Video file not found at {video_path}")
            return

    print(f"\nLoading video from: {video_path}")
    video_frames = load_video_frames_from_file(video_path)
    print(f"Original video shape: {video_frames.shape}")
    # preprocess
    print("Preprocessing video")
    preprocessed_frames = preprocess_video_frames(
        video_frames,
        target_num_frames=int(model_config.get("target_frame_count", 40)),
        target_height=int(model_config.get("target_frame_height", 48)),
        target_width=int(model_config.get("target_frame_width", 48)))
    print(f"Preprocessed shape: {preprocessed_frames.shape}")
    print("\nRunning inference")
    predictions = predict_word(
        model,
        preprocessed_frames,
        index_to_word,
        device,
        args.top_k)
    print("\n" + "=" * 60)
    print("Predictions")
    print("=" * 60)
    for rank, (word, confidence) in enumerate(predictions, 1):
        print(f"{rank}. {word:15s} - {confidence:6.2f}%")
    print("=" * 60)
    print(
        f"\nMost likely word: '{predictions[0][0]}' "
        f"({predictions[0][1]:.2f}% confidence)")


if __name__ == "__main__":
    main()
