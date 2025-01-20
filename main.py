import cv2
import os

def process_video(input_path, output_path):
    # Открываем входной файл
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Не удалось открыть файл: {input_path}")
        return

    # Получаем параметры исходного видео
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Создаём объект VideoWriter для записи выходного видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Можно использовать и другие кодеки
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            # Закончились кадры — выходим из цикла
            break

        # Переводим кадр в градации серого, чтобы найти максимум яркости
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)

        # Рисуем отметку на самом "горячем" (ярком) участке
        cv2.circle(frame, max_loc, 8, (0, 0, 255), 2)

        # Выводим значение яркости
        text = f"Max Intensity: {max_val:.2f}"
        cv2.putText(frame, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Записываем обработанный кадр в выходной файл
        out.write(frame)

    # Освобождаем ресурсы
    cap.release()
    out.release()
    print(f"Видео сохранено в: {output_path}")

if __name__ == "__main__":
    # Пути к входному и выходному файлам
    input_file = "input/2.mp4"
    # Итоговое видео запишем под именем 1_processed.mp4
    output_file = "output/2_processed.mp4"

    # Убеждаемся, что папки существуют (при необходимости можно создать их)
    # os.makedirs("input", exist_ok=True)
    # os.makedirs("output", exist_ok=True)

    process_video(input_file, output_file)
