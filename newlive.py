import cv2
import numpy as np

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

def processFrame(frame: np.ndarray) -> np.ndarray:
    # Обрезаем нижние 10 пикселей
    if frame.shape[0] > 10:
        frame = frame[:-10, :]
    else:
        print("Размер кадра слишком мал для обрезки нижних 10 пикселей")
    
    # Проверка, что ширина кадра достаточная для разбиения
    if frame.shape[1] < 1280:
        print(f"Warning: ширина кадра меньше 1280. Текущая ширина: {frame.shape[1]}")
        return frame

    # Разбиваем кадр на две половины и извлекаем зеленый канал
    left_lo = frame[:, :640, 1].astype(np.uint16)
    right_hi = frame[:, 640:, 1].astype(np.uint16)
    
    # Объединяем две половины: левая – младшие биты, правая – старшие
    combined_16 = (right_hi << 8) | left_lo
    combined_8 = (combined_16 >> 8).astype(np.uint8)
    
    # Инвертируем изображение (белый соответствует высоким температурам)
    inverted = 255 - combined_8

    # Преобразуем инвертированное изображение в формат BGR для дальнейшей отрисовки
    processed = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
    
    # Переводим processed в оттенки серого для поиска яркого пикселя
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # Находим наибольший по яркости пиксель
    _, maxVal, _, maxLoc = cv2.minMaxLoc(gray)
    
    # Определяем размер рамки (например, 20x20 пикселей) вокруг найденного пикселя
    rect_size = 20
    x, y = maxLoc
    top_left = (max(0, x - rect_size // 2), max(0, y - rect_size // 2))
    bottom_right = (min(processed.shape[1]-1, x + rect_size // 2), min(processed.shape[0]-1, y + rect_size // 2))
    
    # Выделяем зеленой рамкой область вокруг самого яркого пикселя
    cv2.rectangle(processed, top_left, bottom_right, (0, 255, 0), 2)
    
    return processed

def main():
    camera_index = selectCamera()
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    # Устанавливаем разрешение и FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Нажмите 'q' для выхода.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр.")
            break
        
        processed_frame = processFrame(frame)
        
        cv2.imshow('original', frame)
        cv2.imshow('processed', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
