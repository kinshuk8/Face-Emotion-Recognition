from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )

    def get_generators(self):
        train_generator = self.datagen.flow_from_directory(
            f'{self.data_dir}/train',
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=32,
            class_mode='categorical'
        )

        test_generator = self.datagen.flow_from_directory(
            f'{self.data_dir}/test',
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=32,
            class_mode='categorical'
        )

        return train_generator, test_generator
