import MySQLdb
from flask import Flask, jsonify, render_template, Response
from flask_mysqldb import MySQL
import cv2
import face_recognition
import numpy as np
import pickle
import easyocr
from datetime import datetime
import os

app = Flask(__name__)

# Koneksi Database MySql
mydb = MySQLdb.connect(
    host="localhost",
    user="root",
    password="",
    database="portal"
)
mycursor = mydb.cursor()

width = 640    
height = 480

model = "static/haarcascade_russian_plate_number.xml"
min_area = 500
count = 0
plate_cascade = cv2.CascadeClassifier(model)
reader = easyocr.Reader(['en'])

detect_plate = {
    'frame' : '',
    'plate' : '',
}

detect_face = {
    'frame': '',
    'nama' : '',
    'nim' : '',
}

known_face_encodings = []
known_face_names = []
ref_dictt = None  

fname = 'static/ref_name.pkl'
with open(fname, 'rb') as fn:
    ref_dictt = pickle.load(fn)

fembed = 'static/ref_embed.pkl'
with open(fembed, 'rb') as fe:
    embed_dictt = pickle.load(fe)

for ref_id, embed_list in embed_dictt.items():
    for my_embed in embed_list:
        known_face_encodings += [my_embed]
        known_face_names += [ref_id]


def face_detect(frame):
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            nim = "N/A"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                ref_id = known_face_names[best_match_index]
                if ref_id in ref_dictt:
                    name = ref_dictt[ref_id].get("name", "Unknown")
                    nim = ref_dictt[ref_id].get("nim", "N/A")
            face_names.append((name, nim))
            detect_face['nama'] = name
            detect_face['nim'] = nim

    process_this_frame = not process_this_frame

    for (top_s, right, bottom, left), (name, nim) in zip(face_locations, face_names):
        top_s *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top_s), (right, bottom), (0, 255, 0), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, f"{name}", (left + 6, bottom - 36), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"{nim}", (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

def anpr(frame):
    global is_plate
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
        
            plate = ''
            for (bbox, text, prob) in results:
                plate += text + " "
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                # cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            detect_plate['plate'] = plate

def generate_frames1():

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        success, frame = camera.read()
        if not success:
            break
        detect_plate['frame'] = frame.copy()

        anpr(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        detect_plate['frame'] = frame

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_frames2():

    camera = cv2.VideoCapture(1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        detect_face['frame'] = frame.copy()

        face_detect(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        detect_face['frame'] = frame

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    sql = 'SELECT gambar_wajah, gambar_plat FROM history ORDER BY id DESC LIMIT 2'
    mycursor.execute(sql)
    result = mycursor.fetchall()
    image_paths = [(row[0], row[1]) for row in result]
    return render_template('index.html', image_paths=image_paths)

@app.route('/save', methods=['POST'])
def save():
    current_time = datetime.now()

    image_wajah = f'{detect_face["nim"]}_{current_time.strftime("%Y_%m_%d_%H_%M_%S")}.png'
    image_wajah = image_wajah.replace(':', '').replace('_', '').replace(' ', '').replace('-', '')
    image_wajah_path = os.path.join('static/imagewajah', image_wajah)
    cv2.imwrite(image_wajah_path, detect_face['frame'])

    image_plat = f'{detect_plate["plate"]}_{current_time.strftime("%Y_%m_%d_%H_%M_%S")}.png'
    image_plat = image_plat.replace(':', '').replace('_', '').replace(' ', '').replace('-', '')
    image_plat_path = os.path.join('static/imageplat', image_plat)
    cv2.imwrite(image_plat_path, detect_plate['frame'])

    query = 'INSERT INTO history (nama, nim, plat, gambar_wajah, gambar_plat) VALUES ("{}","{}","{}","{}","{}")'.format(
        detect_face['nama'],
        detect_face['nim'],
        detect_plate['plate'],
        image_wajah,
        image_plat
    )
    mycursor.execute(query)
    mydb.commit()
    # return jsonify({'message': True})
    img_wajah_url = f'/static/imagewajah/{image_wajah}'
    img_plat_url = f'/static/imageplat/{image_plat}'

    return jsonify({'message': True, 'imgWajahURL': img_wajah_url, 'imgPlatURL': img_plat_url})

@app.route('/video_feed1')
def video_feed1():
    return Response(
            generate_frames1(), 
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

@app.route('/video_feed2')
def video_feed2():
    return Response(
            generate_frames2(), 
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

@app.route('/history')
def history():
    container = []
    sql = 'SELECT * FROM history'
    mycursor.execute(sql)
    results = mycursor.fetchall()
    for data in results:
        container.append(data)
    return render_template('history.html', container=container)

if __name__ == '__main__':
    app.run(debug=True)
