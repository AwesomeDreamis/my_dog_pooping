import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Пути к вашим данным
train_dir = 'data/'

# Используем генератор изображений для автоматической аугментации и загрузки данных
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # используем 20% данных для валидации
)

# Размер изображений и размер батча
img_size = (224, 224)
batch_size = 32

# Подготовка генераторов данных для обучения и валидации
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',  # так как у нас два класса - собаки и не собаки
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

# Загружаем MobileNetV2 без верхней части и добавляем свои слои
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Замораживаем слои базовой модели
base_model.trainable = False

# Создаем новую модель на основе MobileNetV2
model = Sequential([
    base_model,
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # используем сигмоидальную функцию активации для бинарной классификации
])

# Компилируем модель
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Обучаем модель
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size
)

# Оцениваем модель на валидационном наборе данных
val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.n // batch_size)
print(f'Validation accuracy: {val_accuracy*100:.2f}%')

# Сохраняем модель
model.save('my_dog_classifier.h5')

# Визуализируем историю обучения
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over epochs')

plt.show()