import os
from src.train import train_model
from src.real_time_detection import EmotionDetector


def main():
    model_path = 'emotion_model.h5'

    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        detector = EmotionDetector(model_path)
    else:
        print("Training model...")
        data_dir = 'dataset'  # Update with your dataset path
        model, history = train_model(data_dir)
        detector = EmotionDetector(model_path)

    # Real-time detection
    detector.detect_emotions()


if __name__ == '__main__':
    main()
