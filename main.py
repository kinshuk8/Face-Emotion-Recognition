from src.train import train_model
from src.real_time_detection import EmotionDetector


def main():
    # Train model
    data_dir = 'dataset'  # Update with your dataset path
    model, history = train_model(data_dir)

    # Real-time detection
    detector = EmotionDetector('emotion_model.h5')
    detector.detect_emotions()


if __name__ == '__main__':
    main()