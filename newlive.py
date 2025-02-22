import cv2
import numpy as np
import tkinter as tk
import time

# Глобальная переменная для хранения усреднённой (сглаженной) позиции объекта
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

def processFrame(frame: np.ndarray, rect_size, brightness_threshold, frames_avg, max_jump) -> np.ndarray:
    global smoothed_position

    # Обрезаем нижние 10 пикселей
    if frame.shape[0] > 10:
        frame = frame[:-10, :]
    else:
        print("Размер кадра слишком мал для обрезки нижних 10 пикселей")
    
    # Проверяем, что ширина кадра достаточная для разбиения
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
    
    # Для обнаружения яркого объекта переводим processed в оттенки серого
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # Находим ярчайший пиксель
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray)
    if maxVal < brightness_threshold:
        # Если яркость слишком низкая – не обновляем трекер
        candidate = None
    else:
        candidate = maxLoc

    if candidate is not None:
        # Если трекер ещё не инициализирован, задаём его равным найденной точке
        if smoothed_position is None:
            smoothed_position = candidate
        else:
            # Вычисляем разницу между новой точкой и предыдущим усреднённым положением
            dx = candidate[0] - smoothed_position[0]
            dy = candidate[1] - smoothed_position[1]
            distance = np.hypot(dx, dy)
            # Ограничиваем максимальное перемещение
            if distance > max_jump:
                scale = max_jump / distance
                dx = int(dx * scale)
                dy = int(dy * scale)
                candidate = (smoothed_position[0] + dx, smoothed_position[1] + dy)
            # Экспоненциальное сглаживание: alpha = 1/frames_avg
            alpha = 1.0 / frames_avg if frames_avg > 0 else 1.0
            smoothed_position = (int(smoothed_position[0] + alpha * (candidate[0] - smoothed_position[0])),
                                 int(smoothed_position[1] + alpha * (candidate[1] - smoothed_position[1])))
    
    # Если трекер определён, рисуем вокруг него рамку
    if smoothed_position is not None:
        x, y = smoothed_position
        top_left = (max(0, x - rect_size // 2), max(0, y - rect_size // 2))
        bottom_right = (min(processed.shape[1]-1, x + rect_size // 2), min(processed.shape[0]-1, y + rect_size // 2))
        cv2.rectangle(processed, top_left, bottom_right, (0, 255, 0), 2)
    
    return processed

def create_settings_window():
    """
    Создаёт окно с настройками обнаружения объектов с помощью Tkinter.
    Возвращает ссылку на окно и элементы Scale для управления параметрами.
    """
    root = tk.Tk()
    root.title("Настройки обнаружителя объектов")

    # Настройка размера рамки
    tk.Label(root, text="Размер рамки (px):").grid(row=0, column=0, sticky="w")
    rect_size_scale = tk.Scale(root, from_=5, to=100, orient=tk.HORIZONTAL)
    rect_size_scale.set(20)
    rect_size_scale.grid(row=0, column=1)

    # Настройка порога яркости для обнаружения
    tk.Label(root, text="Порог яркости:").grid(row=1, column=0, sticky="w")
    brightness_threshold_scale = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL)
    brightness_threshold_scale.set(200)
    brightness_threshold_scale.grid(row=1, column=1)

    # Настройка количества кадров для усреднения (сглаживание)
    tk.Label(root, text="Кадров для усреднения:").grid(row=2, column=0, sticky="w")
    frames_avg_scale = tk.Scale(root, from_=1, to=30, orient=tk.HORIZONTAL)
    frames_avg_scale.set(5)
    frames_avg_scale.grid(row=2, column=1)

    # Настройка максимального смещения (px)
    tk.Label(root, text="Максимальное смещение (px):").grid(row=3, column=0, sticky="w")
    max_jump_scale = tk.Scale(root, from_=1, to=100, orient=tk.HORIZONTAL)
    max_jump_scale.set(50)
    max_jump_scale.grid(row=3, column=1)

    return root, rect_size_scale, brightness_threshold_scale, frames_avg_scale, max_jump_scale

def main():
    global smoothed_position
    # Создаём окно настроек
    root, rect_size_scale, brightness_threshold_scale, frames_avg_scale, max_jump_scale = create_settings_window()

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

    # Сбрасываем усреднённую позицию при запуске
    smoothed_position = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр.")
            break

        # Получаем текущие значения параметров из окна Tkinter
        rect_size = rect_size_scale.get()
        brightness_threshold = brightness_threshold_scale.get()
        frames_avg = frames_avg_scale.get()
        max_jump = max_jump_scale.get()

        # Обрабатываем кадр с учётом настроек обнаружения и трекинга
        processed_frame = processFrame(frame, rect_size, brightness_threshold, frames_avg, max_jump)

        # Вычисляем FPS
        current_time = time.time()
        fps = 1.0 / (current_time - previous_time) if current_time != previous_time else 0.0
        previous_time = current_time

        # Вывод FPS на изображении (в левом верхнем углу, зеленым цветом)
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Отображаем только итоговое видео
        cv2.imshow('processed', processed_frame)

        # Обновляем окно настроек Tkinter
        root.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

if __name__ == "__main__":
    main()
