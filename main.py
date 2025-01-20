import cv2

def main():
    # Ссылка на поток RTSP. 
    # Учтите, что некоторые Axis-камеры могут использовать slightly иные параметры в URL.
    rtsp_url = "rtsp://root:root@192.168.0.90/axis-media/media.amp?compression=100"

    # Создаём объект захвата видеопотока
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Не удалось открыть поток:", rtsp_url)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось считать кадр")
            break

        # Переводим в градации серого, так как ищем максимум яркости
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Находим минимальное, максимальное значение и их координаты
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray)

        # Рисуем окружность (или квадрат) в координатах самого горячего пикселя maxLoc
        # Для окружности:
        cv2.circle(frame, maxLoc, 8, (0, 0, 255), 2)
        
        # Если хотите квадрат вместо круга:
        # size = 10
        # cv2.rectangle(frame,
        #               (maxLoc[0] - size, maxLoc[1] - size),
        #               (maxLoc[0] + size, maxLoc[1] + size),
        #               (0, 0, 255), 2)

        # Текст можно дополнительно выводить: 
        text = f"Max Intensity: {maxVal:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        # Отображаем результат
        cv2.imshow("Thermal Stream", frame)

        # Нажмите 'q', чтобы выйти из цикла
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
