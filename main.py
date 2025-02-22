import cv2
import numpy as np
import os

# --------------------------------------------
# ПАРАМЕТРЫ ДЛЯ НАСТРОЙКИ
# --------------------------------------------
MIN_MAX_DIFF_THRESHOLD = 20         # Если (maxVal - minVal) < 20 => нет заметного перепада температур
threshold_ratio = 0.8               # Порог для бинаризации (80% от maxVal)
MAX_BBOX_AREA = 50000               # Максимально допустимая площадь bbox ?
ABRUPT_CHANGE_FACTOR = 2.5          # Если площадь меняется более чем в 2.5 раза за кадр - сбой
STABLE_MAX_FRAMES = 50              # Если bbox стабилен ~ 50 кадров => реинициализация
STABLE_POS_THRESH = 10              # Порог для изменения координат центра
STABLE_SIZE_THRESH = 10             # Порог для изменения w/h
# --------------------------------------------

def create_tracker():
    return cv2.legacy.TrackerCSRT_create()

def distance_rects(r1, r2):
    """Евклидово расстояние между центрами (x, y, w, h)."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    c1 = (x1 + w1/2, y1 + h1/2)
    c2 = (x2 + w2/2, y2 + h2/2)
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1])

def unify_bboxes(bboxes, dist_threshold=50):
    """
    Объединяет все близкие bbox'ы (по расстоянию между центрами) в один
    Возвращает список «кластеров» (bbox)
    """
    if not bboxes:
        return []

    merged = []
    used = [False] * len(bboxes)

    for i in range(len(bboxes)):
        if used[i]:
            continue

        cluster = [bboxes[i]]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(bboxes)):
                if not used[j]:
                    for cb in cluster:
                        if distance_rects(cb, bboxes[j]) < dist_threshold:
                            cluster.append(bboxes[j])
                            used[j] = True
                            changed = True
                            break

        # Объединяем bbox’ы кластера в один общий bbox
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

def detect_hot_bbox(frame):
    """
    1. Проверяем разницу minVal и maxVal в кадре (gray), если < MIN_MAX_DIFF_THRESHOLD => нет цели
    2. Порог по threshold_ratio (80%) от maxVal => бинаризуем => контуры
    3. Объединяем близкие контуры => выбираем единственный bbox:
       - если один кластер => его и берём
       - если несколько => выбираем тот, где контур с самой большой средней яркостью
    4. Если bbox > MAX_BBOX_AREA => сбой => None

    Возвращает (x, y, w, h) или None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    minVal, maxVal, _, _ = cv2.minMaxLoc(gray)

    # 1) Проверка, есть ли вообще заметный перепад температур
    if (maxVal - minVal) < MIN_MAX_DIFF_THRESHOLD:
        return None

    # 2) Порог
    thr_val = int(maxVal * threshold_ratio)
    _, mask = cv2.threshold(gray, thr_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Собираем bounding boxes и их среднюю яркость
    boxes = []
    brightness_map = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = gray[y:y+h, x:x+w]
        avg_brightness = roi.mean() if roi.size else 0
        boxes.append((x, y, w, h))
        brightness_map.append(avg_brightness)

    # Объединяем близкие
    merged = unify_bboxes(boxes, dist_threshold=50)
    if len(merged) == 0:
        return None
    if len(merged) == 1:
        # единственный кластер
        (mx, my, mw, mh) = merged[0]
        if mw * mh > MAX_BBOX_AREA:
            return None
        return merged[0]

    # Иначе несколько кластеров - выбираем тот, где контур с самой большой яркостью
    max_idx = np.argmax(brightness_map)
    chosen_box = boxes[max_idx]  # (x, y, w, h)
    x_c, y_c, w_c, h_c = chosen_box
    center_c = (x_c + w_c/2, y_c + h_c/2)

    for mb in merged:
        mx, my, mw, mh = mb
        if (mx <= center_c[0] <= mx + mw) and (my <= center_c[1] <= my + mh):
            if mw * mh > MAX_BBOX_AREA:
                return None
            return mb

    return None

def bbox_area(bbox):
    x, y, w, h = bbox
    return w * h

def is_bbox_change_abrupt(old_bbox, new_bbox):
    """
    Проверяем, не слишком ли резко изменился размер bbox:
    если площадь выросла/уменьшилась более, чем в ABRUPT_CHANGE_FACTOR раз за один кадр => сбой
    """
    old_area = bbox_area(old_bbox)
    new_area = bbox_area(new_bbox)
    if old_area == 0 or new_area == 0:
        return False  # на всякий случай, если вдруг 0, не триггерим
    ratio = max(new_area, old_area) / min(new_area, old_area)
    return ratio > ABRUPT_CHANGE_FACTOR

def is_bbox_almost_same(b1, b2):
    """
    Проверяем, что b1 и b2 "почти одинаковы": 
    - координаты центра отличаются не более STABLE_POS_THRESH 
    - ширина/высота отличаются не более STABLE_SIZE_THRESH
    """
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    cx1, cy1 = (x1 + w1/2), (y1 + h1/2)
    cx2, cy2 = (x2 + w2/2), (y2 + h2/2)

    if abs(cx1 - cx2) > STABLE_POS_THRESH: return False
    if abs(cy1 - cy2) > STABLE_POS_THRESH: return False
    if abs(w1 - w2) > STABLE_SIZE_THRESH: return False
    if abs(h1 - h2) > STABLE_SIZE_THRESH: return False

    return True

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Не удалось открыть: {input_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = create_tracker()
    has_target = False   # Есть ли сейчас зафиксированная цель
    current_bbox = None

    stable_frames = 0    # счётчик, сколько кадров подряд bbox "почти не менялся"
    last_bbox = None     # хранит bbox предыдущего кадра (для стабильности)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not has_target:
            # Пытаемся найти новый bbox на этом кадре
            detected_bbox = detect_hot_bbox(frame)
            if detected_bbox is not None:
                # Инициализируем трекер
                tracker = create_tracker()
                tracker.init(frame, detected_bbox)
                current_bbox = detected_bbox
                has_target = True
                stable_frames = 0
                last_bbox = detected_bbox
            # Записываем кадр (без рамки, потому что цели нет)
            out.write(frame)
        else:
            # Шаг трекера
            success, tracked_box = tracker.update(frame)
            if success:
                # Трекер дал нам bbox
                x_t, y_t, w_t, h_t = [int(v) for v in tracked_box]
                new_bbox = (x_t, y_t, w_t, h_t)

                # Для точности: ищем горячую зону снова
                detected_bbox = detect_hot_bbox(frame)
                if detected_bbox is not None:
                    # Проверим резкий скачок размера
                    if not is_bbox_change_abrupt(new_bbox, detected_bbox):
                        # Если нет резкого скачка, берём detected_bbox
                        if bbox_area(detected_bbox) <= MAX_BBOX_AREA:
                            new_bbox = detected_bbox
                            # Обновим трекер
                            tracker = create_tracker()
                            tracker.init(frame, new_bbox)

                # Проверяем, не слишком ли большой bbox
                if bbox_area(new_bbox) > MAX_BBOX_AREA:
                    # Считаем, что цели нет
                    has_target = False
                    current_bbox = None
                    out.write(frame)
                    continue

                # Проверяем, не резко ли изменился bbox (по сравнению с current_bbox)
                if current_bbox is not None:
                    if is_bbox_change_abrupt(current_bbox, new_bbox):
                        # Резкий скачок => сбрасываем цель
                        has_target = False
                        current_bbox = None
                        out.write(frame)
                        continue

                current_bbox = new_bbox

                # Проверяем "стабильность" — если bbox почти не меняется, увеличиваем счётчик
                if last_bbox is not None and is_bbox_almost_same(last_bbox, current_bbox):
                    stable_frames += 1
                else:
                    stable_frames = 0

                # Если bbox "завис" слишком надолго, сбрасываем цель
                if stable_frames > STABLE_MAX_FRAMES:
                    has_target = False
                    current_bbox = None
                    out.write(frame)
                    continue

                last_bbox = current_bbox

                # Рисуем итоговый bbox
                x, y, w, h = current_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Пример подписи: координаты центра bbox
                cx = x + w//2
                cy = y + h//2
                text = f"BBox: ({x},{y},{w},{h}), center=({cx},{cy})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # Трекер "потерял" цель
                has_target = False
                current_bbox = None

            out.write(frame)

    cap.release()
    out.release()
    print(f"Результат сохранён в: {output_path}")

if __name__ == "__main__":
    # Три варианта:
    print("Выберите режим:")
    print("1) Использовать RTSP-поток (rtsp://root:root@192.168.0.90/axis-media/media.amp?compression=100)")
    print("2) Использовать локальное видео (input/1.mp4)")
    print("3) Использовать HTTP (http://root:root@192.168.0.90/axis-cgi/mjpg/video.cgi)")
    mode = input("Ваш выбор (1, 2 или 3): ").strip()

    if mode == '1':
        input_source = "rtsp://root:root@192.168.0.90/axis-media/media.amp?compression=100"
    elif mode == '3':
        # ?fps=25&resolution=640x480
        input_source = "http://root:root@192.168.0.90/axis-cgi/mjpg/video.cgi"
    else:
        input_source = "input/3.mp4"

    os.makedirs("output", exist_ok=True)
    output_file = "output/1_processed.mp4"

    process_video(input_source, output_file)
