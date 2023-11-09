import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2, decode_predictions


# Загружаем предобученную модель MobileNetV2
model = MobileNetV2(weights='imagenet')
# model = load_model('путь_к_вашей_сохраненной_модели.h5')


def recognize_image_loadmodel(image_path):
    # Загрузка и предобработка изображения
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))  # MobileNetV2 ожидает изображения размером 224x224
    image_array = np.expand_dims(image_resized, axis=0)
    image_processed = preprocess_input(image_array)
    # Делаем предсказание
    predictions = model.predict(image_processed)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    print(decoded_predictions)
    for _, label, prob in decoded_predictions:
        if label == 'Labrador_retriever':
            print(f"Found a Labrador Retriever with probability: {prob}")
        else:
            print(f"Detected: {label} with probability: {prob}")


def recognize_image_mymodel(image_path):
    # Загрузка изображения и преобразование его в массив
    img = image.load_img(image_path, target_size=(150, 150))  # Используйте те же размеры, что и при обучении
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Создание пакета из одного изображения
    # Нормализация изображения
    img_array /= 255.0
    # Предсказание
    prediction = model.predict(img_array)
    if prediction < 0.5:
        print(f"Это собака, которая сидит! Вероятность: {1 - prediction[0, 0]:.2f}")
    else:
        print(f"Это не собака, которая сидит! Вероятность: {prediction[0, 0]:.2f}")
    return prediction


def recognize_from_camera_loadmodel():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Масштабируем и предварительно обрабатываем кадр перед предсказанием
        resized_frame = cv2.resize(frame, (224, 224))
        frame_array = np.expand_dims(resized_frame, axis=0)
        frame_processed = preprocess_input(frame_array)
        # Делаем предсказание
        predictions = model.predict(frame_processed)
        decoded_predictions = decode_predictions(predictions, top=1)[0]
        for _, label, prob in decoded_predictions:
            cv2.putText(frame, f"{label}: {prob:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Отображаем результаты
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def recognize_from_camera_mymodel():
    cap = cv2.VideoCapture(0)
    while True:
        # Чтение кадра из видеопотока
        ret, frame = cap.read()
        if not ret:
            break
        # Предсказание текущего кадра
        prediction = recognize_image_mymodel(frame)
        # Вывод результата предсказания на экран
        if prediction < 0.5:
            text = "Сидящая собака"
        else:
            text = "Не сидящая собака"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Показать кадр
        cv2.imshow('Video', frame)
        # Выход из цикла по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


# Выберите один из вариантов:
# recognize_image_loadmodel('Ris.6-SHustryj-korgi.jpg')  # Для распознавания собак на изображении
recognize_from_camera_mymodel()  # Для распознавания в реальном времени с веб-камеры
