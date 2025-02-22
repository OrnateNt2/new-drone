import cv2
import numpy as np
import tkinter as tk
import time

# Глобальная переменная для хранения усреднённой позиции (для "Smoothed Tracker")
smoothed_position = None

def listAvailableCameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def selectCamera():
    cameras = listAvailableCameras()
    if not cameras:
        print("Нет доступных камер")
        exit(0)
    
    print("Доступные камеры:")
    for index in cameras:
        print(f"[{index}] Камера {index}")
    
    while True:
        try:
            selected_index = int(input("Выберите номер камеры: "))
            if selected_index in cameras:
                return selected_index
            else:
                print("Неверный выбор. Попробуйте снова")
        except ValueError:
            print("Введите номер камеры")

def detectBrightSpots(gray, detection_count, min_distance, brightness_threshold):
    """
    Находит несколько ярких точек в изображении gray.
    После каждого обнаружения область вокруг точки (радиус = min_distance) обнуляется,
    чтобы следующие обнаружения были далеко от предыдущих.
    """
    spots = []
    temp = gray.copy()
    for i in range(detection_count):
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(temp)
        if maxVal < brightness_threshold:
            break
        spots.append(maxLoc)
        # Обнуляем область вокруг найденной точки
        cv2.circle(temp, maxLoc, min_distance, 0, thickness=-1)
    return spots

def processFrame(frame: np.ndarray, tracker_type: str, rect_size: int, brightness_threshold: int,
                 frames_avg: int, max_jump: int, detection_count: int, min_distance: int) -> np.ndarray:
    global smoothed_position

    # Обрезаем нижние 10 пикселей
    if frame.shape[0] > 10:
        frame = frame[:-10, :]
    else:
        print("Размер кадра слишком мал для обрезки нижних 10 пикселей")
    
    # Проверяем, что ширина кадра достаточная для разбиения на две части
    if frame.shape[1] < 1280:
        print(f"Warning: ширина кадра меньше 1280. Текущая ширина: {frame.shape[1]}")
        return frame

    # Разбиваем кадр на две половины и извлекаем зеленый канал
    left_lo = frame[:, :640, 1].astype(np.uint16)
    right_hi = frame[:, 640:, 1].astype(np.uint16)
    
    # Объединяем две половины: левая – младшие биты, правая – старшие
    combined_16 = (right_hi << 8) | left_lo
    combined_8 = (combined_16 >> 8).astype(np.uint8)
    
    # Инвертируем изображение (белый соответствует более высоким температурам)
    inverted = 255 - combined_8

    # Преобразуем в формат BGR для возможности рисования цветных рамок
    processed = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
    
    # Переводим processed в оттенки серого для анализа яркости
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    if tracker_type == "Brightest Pixel":
        # Простой трекер: выбирается самый яркий пиксель без сглаживания
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray)
        if maxVal >= brightness_threshold:
            x, y = maxLoc
            top_left = (max(0, x - rect_size // 2), max(0, y - rect_size // 2))
            bottom_right = (min(processed.shape[1]-1, x + rect_size // 2), min(processed.shape[0]-1, y + rect_size // 2))
            cv2.rectangle(processed, top_left, bottom_right, (0, 255, 0), 2)
    
    elif tracker_type == "Smoothed Tracker":
        # Трекер с экспоненциальным сглаживанием
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray)
        candidate = maxLoc if maxVal >= brightness_threshold else None
        
        if candidate is not None:
            if smoothed_position is None:
                smoothed_position = candidate
            else:
                dx = candidate[0] - smoothed_position[0]
                dy = candidate[1] - smoothed_position[1]
                distance = np.hypot(dx, dy)
                if distance > max_jump:
                    scale = max_jump / distance
                    dx = int(dx * scale)
                    dy = int(dy * scale)
                    candidate = (smoothed_position[0] + dx, smoothed_position[1] + dy)
                alpha = 1.0 / frames_avg if frames_avg > 0 else 1.0
                smoothed_position = (int(smoothed_position[0] + alpha * (candidate[0] - smoothed_position[0])),
                                     int(smoothed_position[1] + alpha * (candidate[1] - smoothed_position[1])))
        if smoothed_position is not None:
            x, y = smoothed_position
            top_left = (max(0, x - rect_size // 2), max(0, y - rect_size // 2))
            bottom_right = (min(processed.shape[1]-1, x + rect_size // 2), min(processed.shape[0]-1, y + rect_size // 2))
            cv2.rectangle(processed, top_left, bottom_right, (0, 255, 0), 2)
    
    elif tracker_type == "Centroid Tracker":
        # Вычисляем центроид нескольких ярких точек
        spots = detectBrightSpots(gray, detection_count, min_distance, brightness_threshold)
        if spots:
            avg_x = int(np.mean([pt[0] for pt in spots]))
            avg_y = int(np.mean([pt[1] for pt in spots]))
            candidate = (avg_x, avg_y)
            top_left = (max(0, candidate[0] - rect_size // 2), max(0, candidate[1] - rect_size // 2))
            bottom_right = (min(processed.shape[1]-1, candidate[0] + rect_size // 2), min(processed.shape[0]-1, candidate[1] + rect_size // 2))
            cv2.rectangle(processed, top_left, bottom_right, (0, 255, 0), 2)
    
    elif tracker_type == "Multiple Spots":
        # Отмечаем каждую обнаруженную яркую точку рамкой
        spots = detectBrightSpots(gray, detection_count, min_distance, brightness_threshold)
        for (x, y) in spots:
            top_left = (max(0, x - rect_size // 2), max(0, y - rect_size // 2))
            bottom_right = (min(processed.shape[1]-1, x + rect_size // 2), min(processed.shape[0]-1, y + rect_size // 2))
            cv2.rectangle(processed, top_left, bottom_right, (0, 255, 0), 2)
    
    return processed

def create_settings_window():
    """
    Создаёт окно с настройками обнаружения объектов с помощью Tkinter.
    Возвращает ссылку на окно и элементы управления параметрами.
    """
    root = tk.Tk()
    root.title("Настройки обнаружителя объектов")

    # Размер рамки
    tk.Label(root, text="Размер рамки (px):").grid(row=0, column=0, sticky="w")
    rect_size_scale = tk.Scale(root, from_=5, to=100, orient=tk.HORIZONTAL)
    rect_size_scale.set(20)
    rect_size_scale.grid(row=0, column=1)

    # Порог яркости
    tk.Label(root, text="Порог яркости:").grid(row=1, column=0, sticky="w")
    brightness_threshold_scale = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL)
    brightness_threshold_scale.set(200)
    brightness_threshold_scale.grid(row=1, column=1)

    # Количество кадров для усреднения (сглаживание)
    tk.Label(root, text="Кадров для усреднения:").grid(row=2, column=0, sticky="w")
    frames_avg_scale = tk.Scale(root, from_=1, to=30, orient=tk.HORIZONTAL)
    frames_avg_scale.set(5)
    frames_avg_scale.grid(row=2, column=1)

    # Максимальное смещение (px)
    tk.Label(root, text="Максимальное смещение (px):").grid(row=3, column=0, sticky="w")
    max_jump_scale = tk.Scale(root, from_=1, to=100, orient=tk.HORIZONTAL)
    max_jump_scale.set(50)
    max_jump_scale.grid(row=3, column=1)

    # Выбор типа трекера
    tracker_options = ["Brightest Pixel", "Smoothed Tracker", "Centroid Tracker", "Multiple Spots"]
    tracker_var = tk.StringVar(root)
    tracker_var.set("Smoothed Tracker")
    tk.Label(root, text="Тип трекера:").grid(row=4, column=0, sticky="w")
    tracker_menu = tk.OptionMenu(root, tracker_var, *tracker_options)
    tracker_menu.grid(row=4, column=1)

    # Количество обнаружений (для трекеров, использующих несколько точек)
    tk.Label(root, text="Количество обнаружений:").grid(row=5, column=0, sticky="w")
    detection_count_scale = tk.Scale(root, from_=1, to=10, orient=tk.HORIZONTAL)
    detection_count_scale.set(3)
    detection_count_scale.grid(row=5, column=1)

    # Минимальное расстояние между обнаружениями (px)
    tk.Label(root, text="Минимальное расстояние (px):").grid(row=6, column=0, sticky="w")
    min_distance_scale = tk.Scale(root, from_=1, to=100, orient=tk.HORIZONTAL)
    min_distance_scale.set(30)
    min_distance_scale.grid(row=6, column=1)

    return (root, rect_size_scale, brightness_threshold_scale, frames_avg_scale,
            max_jump_scale, tracker_var, detection_count_scale, min_distance_scale)

def main():
    global smoothed_position
    # Создаём окно настроек
    (root, rect_size_scale, brightness_threshold_scale, frames_avg_scale,
     max_jump_scale, tracker_var, detection_count_scale, min_distance_scale) = create_settings_window()

    camera_index = selectCamera()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    # Устанавливаем разрешение и FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    previous_time = time.time()
    print("Нажмите 'q' для выхода.")
    smoothed_position = None  # Сброс трекера при запуске

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр.")
            break

        # Считываем текущие значения параметров из окна настроек
        rect_size = rect_size_scale.get()
        brightness_threshold = brightness_threshold_scale.get()
        frames_avg = frames_avg_scale.get()
        max_jump = max_jump_scale.get()
        tracker_type = tracker_var.get()
        detection_count = detection_count_scale.get()
        min_distance = min_distance_scale.get()

        # Обрабатываем кадр с учётом выбранного трекера и параметров
        processed_frame = processFrame(frame, tracker_type, rect_size, brightness_threshold,
                                       frames_avg, max_jump, detection_count, min_distance)

        # Вычисляем FPS
        current_time = time.time()
        fps = 1.0 / (current_time - previous_time) if current_time != previous_time else 0.0
        previous_time = current_time
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Отображаем только итоговое видео
        cv2.imshow('processed', processed_frame)
        root.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

if __name__ == "__main__":
    main()
