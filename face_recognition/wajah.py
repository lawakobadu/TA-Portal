import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt

# Load Cascade Classifier untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load model Anda
model = tf.keras.models.load_model('model_wajah.h5')

x1, y1, x2, y2 = 0, 0, 0, 0

# Fungsi untuk melakukan deteksi wajah
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Ambil wajah pertama yang terdeteksi
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        return (x, y, x + w, y + h), face_roi
    else:
        return None, None

# Fungsi untuk melakukan deteksi objek pada frame kamera
def detect_objects(frame):
    # Lakukan deteksi objek di sini menggunakan model Anda
    # Anda perlu mengubah kode ini sesuai dengan model dan data Anda
    # Misalnya, Anda dapat menggunakan model TensorFlow Lite untuk deteksi objek

    # Ubah frame menjadi bentuk yang dapat digunakan oleh model
    frame_resized = cv2.resize(frame, (40, 40))  # Ganti ukuran frame sesuai dengan model Anda
    x = image.img_to_array(frame_resized)
    x = np.expand_dims(x, axis=0)

    # Lakukan prediksi
    classes = model.predict(x)

    # Interpretasi hasil prediksi (ganti sesuai dengan model Anda)
    class_list = ['Akmal', 'Alif', 'Fadli', 'Fairuz', 'Habib', 'Lukman', 'Radhian']  # Ganti dengan daftar kelas yang sesuai
    predicted_class_index = np.argmax(classes)
    prediction_prob = np.max(classes)

    if prediction_prob < 0.5:
        predicted_class = "Objek tidak terdefinisi"
        bounding_box = None
    else:
        predicted_class = class_list[predicted_class_index]
        bounding_box = (x1, y1, x2, y2)  # Ganti dengan koordinat yang sesuai

    return predicted_class, prediction_prob, bounding_box

# Buka kamera laptop
cap = cv2.VideoCapture(0)  # Gunakan indeks 0 untuk kamera default

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lakukan deteksi wajah pada frame kamera
    face_coords, face_roi = detect_faces(frame)

    if face_coords is not None:
        # Lakukan deteksi objek pada wajah yang terdeteksi
        predicted_class, prediction_prob, bounding_box = detect_objects(face_roi)

        # Tampilkan hasil prediksi pada layar
        cv2.putText(frame, f"Hasil Prediksi: {predicted_class} (Probabilitas: {prediction_prob:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Gambar kotak pembatas pada objek yang diidentifikasi di atas wajah
        if bounding_box is not None:
            x1, y1, x2, y2 = bounding_box
            x1 += face_coords[0]
            y1 += face_coords[1]
            x2 += face_coords[0]
            y2 += face_coords[1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Deteksi Objek', frame)

    # Tekan tombol "q" untuk keluar dari aplikasi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela tampilan
cap.release()
cv2.destroyAllWindows()
