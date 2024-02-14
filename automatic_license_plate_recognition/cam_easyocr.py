import cv2
import easyocr

model = "haarcascade_russian_plate_number.xml"
min_area = 500
cap = cv2.VideoCapture(0)
count = 0

if not cap.isOpened():
    print("Gagal membuka video capture")
    exit()

# Inisialisasi Cascade Classifier di luar loop while untuk mengoptimalkan performa
plate_cascade = cv2.CascadeClassifier(model)
reader = easyocr.Reader(['en'])

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame")
        break

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = frame[y: y + h, x: x + w]

            # Menggunakan EasyOCR untuk mengenali teks dalam ROI
            results = reader.readtext(img_roi)

            for (bbox, text, prob) in results:
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                # cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Hasil/gambar_" + str(count) + ".jpg", img_roi)
        cv2.imshow("Hasil", frame)
        cv2.waitKey(500)
        count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup video capture dan jendela tampilan
cap.release()
cv2.destroyAllWindows()
