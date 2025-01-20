import cv2
import numpy as np
import os

def process_video(input_path, output_path, history_size=5, threshold_ratio=0.8):
    """
    - input_path: путь к исходному видео
    - output_path: путь к выходному видео
    - history_size: сколько предыдущих кадров хранить (для сглаживания)
    - threshold_ratio: во сколько раз порог ниже максимальной яркости
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Не удалось открыть файл: {input_path}")
        return

    # Параметры исходного видео
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Объект для записи выходного видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Можно использовать и другие кодеки
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Хранилище последних кадров (grayscale) для усреднения/сглаживания
    frames_gray_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            # Кадры закончились
            break

        # Преобразуем в градации серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Добавляем текущий кадр в "историю"
        frames_gray_history.append(gray)
        if len(frames_gray_history) > history_size:
            # Удаляем самый старый кадр, если список переполнен
            frames_gray_history.pop(0)

        # Составляем "комбинированное" изображение из последних N кадров,
        # берём максимум яркости по каждому пикселю, чтобы подсветить горячие зоны
        combined_gray = np.maximum.reduce(frames_gray_history)
        combined_max = combined_gray.max()

        # Если вообще нет ярких пикселей, пропускаем порог
        if combined_max < 10:
            # Просто записываем оригинальный кадр, без выделений
            out.write(frame)
            continue

        # Рассчитываем порог (например, 80% от максимума)
        threshold_val = int(combined_max * threshold_ratio)

        # Бинаризация для выделения горячих зон
        _, thresh = cv2.threshold(combined_gray, threshold_val, 255, cv2.THRESH_BINARY)

        # Находим контуры (каждый контур — потенциально отдельная горячая область)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Отрисовываем bounding box для каждого контура
        # И собираем текст со сведениями о координатах
        box_info = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w//2
            cy = y + h//2

            # Рисуем зелёный квадрат (BGR: (0,255,0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Сохраняем информацию о центре и размере
            box_info.append((cx, cy, w, h))

        # Выводим текст слева сверху: для каждого bbox своя строка
        # Если много контуров, строки будут идти друг за другом
        for i, (cx, cy, w, h) in enumerate(box_info):
            text = f"Box{i}: center=({cx},{cy}), size=({w}x{h})"
            cv2.putText(frame, text, (10, 30 + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Сохраняем кадр в выходной файл
        out.write(frame)

    cap.release()
    out.release()
    print(f"Видео сохранено в: {output_path}")

if __name__ == "__main__":
    # Пути к входному и выходному файлам
    input_file = "input/2.mp4"
    output_file = "output/2_processed.mp4"

    # Убедимся, что папки существуют (при необходимости можно создать их)
    # os.makedirs("input", exist_ok=True)
    # os.makedirs("output", exist_ok=True)

    process_video(input_file, output_file)
