import cv2
import numpy as np
import os

# Для OpenCV >= 4.5.2 обычно трекеры находятся в модуле legacy
# Если у вас другая версия, возможно, нужно: cv2.TrackerCSRT_create() вместо legacy
def create_tracker():
    return cv2.legacy.TrackerCSRT_create()

def distance_rects(r1, r2):
    """Вспомогательная функция для оценки "близости" двух прямоугольников.
       Возвращает расстояние между центрами."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    c1 = (x1 + w1/2, y1 + h1/2)
    c2 = (x2 + w2/2, y2 + h2/2)
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1])

def unify_bboxes(bboxes, dist_threshold=50):
    """
    Объединяем все bounding box’ы, которые близки друг к другу, 
    в один общий bbox. "Близки" — если расстояние между центрами
    не превышает dist_threshold (или перекрываются).
    
    Возвращает список bboxes, в которых каждый «кластер» объединён в один.
    Если в итоге получится несколько разрозненных кластеров, 
    значит они действительно далеко друг от друга.
    """
    if not bboxes:
        return []

    merged = []
    used = [False] * len(bboxes)

    for i in range(len(bboxes)):
        if used[i]:
            continue
        # Начинаем новый кластер с i-го bbox
        x, y, w, h = bboxes[i]
        cluster = [(x, y, w, h)]
        used[i] = True

        # Ищем, кто ещё близок к i
        something_merged = True
        while something_merged:
            something_merged = False
            for j in range(len(bboxes)):
                if not used[j]:
                    x2, y2, w2, h2 = bboxes[j]
                    # Проверим расстояние до любого из уже включённых в cluster
                    for (cx, cy, cw, ch) in cluster:
                        if distance_rects((cx, cy, cw, ch), (x2, y2, w2, h2)) < dist_threshold:
                            cluster.append((x2, y2, w2, h2))
                            used[j] = True
                            something_merged = True
                            break

        # Объединяем все bbox из cluster в один общий
        xs = [c[0] for c in cluster]
        ys = [c[1] for c in cluster]
        ws = [c[2] for c in cluster]
        hs = [c[3] for c in cluster]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs[i] + ws[i] for i in range(len(xs)))
        y_max = max(ys[i] + hs[i] for i in range(len(ys)))
        merged.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return merged

def detect_hot_bbox(frame, threshold_ratio=0.8):
    """
    Ищет "горячие" области по яркости (grayscale).
    - Переводит в gray
    - Ставит порог как 80% от max яркости (можно настраивать)
    - Находит контуры
    - Все близкие контуры объединяет
    - Если остаётся несколько сильно разрозненных областей, возвращает:
        - либо одну большую, если они близки
        - либо самую яркую, если далеки друг от друга
    Возвращает либо None, либо кортеж (x, y, w, h).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    max_val = gray.max()
    if max_val < 10:  # Всё слишком тёмное
        return None

    thr_val = int(max_val * threshold_ratio)
    _, mask = cv2.threshold(gray, thr_val, 255, cv2.THRESH_BINARY)

    # Находим контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Собираем bounding boxes
    boxes = []
    brightness_map = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))
        # Оценим среднюю яркость внутри контура, чтобы потом понять, какая ярче
        roi = gray[y:y+h, x:x+w]
        avg_brightness = roi.mean() if roi.size > 0 else 0
        brightness_map.append(avg_brightness)

    # Объединяем все, что рядом
    merged = unify_bboxes(boxes, dist_threshold=50)

    # Если после объединения один bbox - возвращаем его
    if len(merged) == 1:
        return merged[0]

    # Иначе несколько разрозненных объединённых кластеров.
    # По условию: если все далеко, берём самую яркую. 
    # Как понять "самую яркую"? 
    # Можно брать bbox с наибольшей средней яркостью из исходных контуров.
    # Но нужно понять, какой bbox соответствует какой группе. 
    # Для упрощения возьмём bbox, у которого max средняя яркость (из всех изначальных контуров).
    max_idx = np.argmax(brightness_map)
    # И найдём, к какому кластеру относится этот контур.
    chosen_box = boxes[max_idx]  # (x, y, w, h)

    # И найдём тот merged bbox, в котором он сидит
    x_c, y_c, w_c, h_c = chosen_box
    center_c = (x_c + w_c/2, y_c + h_c/2)
    
    best_cluster_bbox = None
    for mb in merged:
        mx, my, mw, mh = mb
        if mx <= center_c[0] <= mx+mw and my <= center_c[1] <= my+mh:
            best_cluster_bbox = mb
            break

    return best_cluster_bbox

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Не удалось открыть файл: {input_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Создаём трекер (CSRT точнее, но тяжелее, чем KCF/MOSSE)
    tracker = create_tracker()
    initialized_tracker = False
    current_bbox = None  # (x, y, w, h)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not initialized_tracker:
            # Первый раз ищем hot bbox и инициализируем трекер
            bbox = detect_hot_bbox(frame)
            if bbox is not None:
                tracker.init(frame, tuple(bbox))
                current_bbox = bbox
                initialized_tracker = True
            # Запишем кадр (пока без bbox, если не нашли)
            out.write(frame)
            continue
        else:
            # Шаг трекера
            success, tracked_bbox = tracker.update(frame)
            if success:
                # tracked_bbox может быть float, приводим к int
                x_t, y_t, w_t, h_t = [int(v) for v in tracked_bbox]
                current_bbox = (x_t, y_t, w_t, h_t)

        # Делаем новую детекцию (чтобы следить за изменениями)
        detected_bbox = detect_hot_bbox(frame)

        # Если ничего не нашли — используем bbox от трекера
        # Если нашли, проверяем "близость" с current_bbox
        if detected_bbox is not None:
            dist = distance_rects(current_bbox, detected_bbox)
            # Если новый bbox не слишком далеко, 
            # обновим трекер на detected_bbox (более точная позиция)
            if dist < 100:  # порог "прыжка" — подбирается эмпирически
                current_bbox = detected_bbox
                tracker = create_tracker()
                tracker.init(frame, tuple(current_bbox))
            else:
                # Иначе считаем, что объект не может так резко перескочить
                # Оставляем tracked_bbox
                pass

        # Рисуем итоговый current_bbox
        if current_bbox is not None:
            x, y, w, h = current_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Сохраняем кадр
        out.write(frame)

    cap.release()
    out.release()
    print(f"Видео сохранено в: {output_path}")

if __name__ == "__main__":
    input_file = "input/2.mp4"
    output_file = "output/2_processed.mp4"
    # os.makedirs("output", exist_ok=True)
    process_video(input_file, output_file)
