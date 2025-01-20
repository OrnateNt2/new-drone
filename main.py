import cv2
import numpy as np
import os

# Для OpenCV >= 4.5.2 трекеры могут быть доступны в модуле legacy
# В некоторых сборках это может выглядеть как: cv2.TrackerCSRT_create()
def create_tracker():
    return cv2.legacy.TrackerCSRT_create()

def distance_rects(r1, r2):
    """
    Возвращает Евклидово расстояние между центрами двух прямоугольников (x, y, w, h).
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    c1 = (x1 + w1/2, y1 + h1/2)
    c2 = (x2 + w2/2, y2 + h2/2)
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1])

def unify_bboxes(bboxes, dist_threshold=50):
    """
    Объединяет все прямоугольники (x, y, w, h), центры которых не дальше dist_threshold, в один общий bbox.
    Возвращает список объединённых bbox’ов (их может быть несколько, если есть сильно отдалённые группы).
    """
    if not bboxes:
        return []

    merged = []
    used = [False] * len(bboxes)

    for i in range(len(bboxes)):
        if used[i]:
            continue
        # Начинаем новый кластер с i-го bbox
        cluster = [bboxes[i]]
        used[i] = True

        # Ищем, кто ещё близок к этому кластеру
        something_merged = True
        while something_merged:
            something_merged = False
            for j in range(len(bboxes)):
                if not used[j]:
                    for cbox in cluster:
                        if distance_rects(cbox, bboxes[j]) < dist_threshold:
                            cluster.append(bboxes[j])
                            used[j] = True
                            something_merged = True
                            break

        # Объединяем прямоугольники в кластере в один общий bbox
        xs = [c[0] for c in cluster]
        ys = [c[1] for c in cluster]
        ws = [c[2] for c in cluster]
        hs = [c[3] for c in cluster]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs[i] + ws[i] for i in range(len(xs)))
        y_max = max(ys[i] + hs[i] for i in range(len(ys)))
        merged_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        merged.append(merged_bbox)

    return merged

def detect_hot_bbox(frame, threshold_ratio=0.8):
    """
    Ищет "горячие" зоны по яркости.
    Возвращает один bbox (x, y, w, h) — либо None, если ничего не найдено.

    Алгоритм:
    1) Перевод кадра в grayscale.
    2) Определение порога как threshold_ratio (напр. 0.8) от максимальной яркости.
    3) Бинаризация, поиск контуров.
    4) Объединение близких контуров в общий bbox.
    5) Если в итоге несколько кластеров, берём тот, где выше средняя яркость.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    max_val = gray.max()
    if max_val < 10:
        # Слишком тёмный кадр, ничего "горячего" нет
        return None

    thr_val = int(max_val * threshold_ratio)
    _, mask = cv2.threshold(gray, thr_val, 255, cv2.THRESH_BINARY)

    # Находим контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Собираем bounding boxes
    boxes = []
    brightness_map = []  # средняя яркость внутри контура
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))

        roi = gray[y:y+h, x:x+w]
        avg_brightness = roi.mean() if roi.size > 0 else 0
        brightness_map.append(avg_brightness)

    # Объединяем близкие
    merged = unify_bboxes(boxes, dist_threshold=50)

    if len(merged) == 1:
        # Всего один общий bbox
        return merged[0]

    # Несколько разрозненных групп. По условию, если далеко друг от друга,
    # выбираем группу, в которой контур с максимальной яркостью.
    max_idx = int(np.argmax(brightness_map))
    chosen_box = boxes[max_idx]  # (x, y, w, h)
    x_c, y_c, w_c, h_c = chosen_box
    center_c = (x_c + w_c/2, y_c + h_c/2)

    # Выясняем, в каком merged bbox находится этот контур
    for mb in merged:
        mx, my, mw, mh = mb
        if (mx <= center_c[0] <= mx + mw) and (my <= center_c[1] <= my + mh):
            return mb

    # Если почему-то не нашли (крайне маловероятно), вернём просто chosen_box
    return chosen_box

def process_video(input_path, output_path):
    """
    Считывает поток/видео из input_path, обрабатывает кадры и записывает результат в output_path.
    Логика:
    1) Ищем "горячую" зону => bbox, инициализируем трекер CSRT (один bbox на всё видео).
    2) На каждом кадре:
       - трекер пытается предсказать новый bbox
       - параллельно ищем новый горячий bbox (detect_hot_bbox)
       - если новый детект близко к трекеру, обновляем трекер (на случай изменения)
       - если далеко, считаем, что объект не мог резко перескочить
    3) Рисуем bbox и слева сверху пишем: "Center=(cx_rel, cy_rel), Size=(w×h)"
       где (cx_rel, cy_rel) — координаты центра bbox относительно центра кадра.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Не удалось открыть: {input_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Если вдруг (например, RTSP) fps не определился корректно, подстрахуемся
    if fps < 1.0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Инициализация трекера
    tracker = create_tracker()
    initialized_tracker = False
    current_bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not initialized_tracker:
            # Ищем горячий bbox на первом "удачном" кадре
            bbox = detect_hot_bbox(frame)
            if bbox is not None:
                tracker.init(frame, bbox)  # tuple(x, y, w, h)
                current_bbox = bbox
                initialized_tracker = True

            # Запишем кадр (если bbox не найден, пока без рамки)
            out.write(frame)
            continue
        else:
            # Шаг трекера
            success, tracked_bbox = tracker.update(frame)
            if success:
                x_t, y_t, w_t, h_t = [int(v) for v in tracked_bbox]
                current_bbox = (x_t, y_t, w_t, h_t)

        # Новый детект
        detected_bbox = detect_hot_bbox(frame)
        if detected_bbox is not None and current_bbox is not None:
            dist = distance_rects(current_bbox, detected_bbox)
            # Если новый bbox не слишком далеко — обновим трекер
            if dist < 100:
                current_bbox = detected_bbox
                tracker = create_tracker()
                tracker.init(frame, detected_bbox)

        # Рисуем итоговый current_bbox
        if current_bbox is not None:
            x, y, w, h = current_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Координаты центра bbox относительно центра кадра
            cx_rel = int((x + w/2) - (width/2))
            cy_rel = int((y + h/2) - (height/2))
            text = f"Center=({cx_rel},{cy_rel}), Size=({w}x{h})"

            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Видео сохранено в: {output_path}")

if __name__ == "__main__":
    # Спрашиваем у пользователя, какой вход использовать
    print("Выберите режим:")
    print("1) Использовать RTSP-поток (rtsp://root:root@192.168.0.90/axis-media/media.amp?compression=100)")
    print("2) Использовать локальное видео (input/1.mp4)")
    mode = input("Ваш выбор (1 или 2): ")

    if mode.strip() == '1':
        input_source = "rtsp://root:root@192.168.0.90/axis-media/media.amp?compression=100"
    else:
        input_source = "input/1.mp4"

    # Убедимся, что каталог output существует
    os.makedirs("output", exist_ok=True)
    output_file = "output/1_processed.mp4"

    process_video(input_source, output_file)
