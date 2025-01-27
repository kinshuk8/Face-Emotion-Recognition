from src.model import create_model
from src.data_loader import DataLoader

def train_model(data_dir, epochs=50):
    # Load data
    data_loader = DataLoader(data_dir)
    train_generator, test_generator = data_loader.get_generators()

    # Create and train model
    model = create_model()

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator
    )

    # Save model
    model.save('emotion_model.h5')
    return model, history
