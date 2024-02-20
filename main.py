import MySQLdb
from flask import Flask, jsonify, render_template, Response, request, url_for, session, redirect
from flask_mysqldb import MySQL
import cv2
import face_recognition
import numpy as np
import pickle
import easyocr, json
from datetime import datetime
import bcrypt, os, base64

app = Flask(__name__)
app.secret_key = 'SPD'

# Koneksi Database MySql
mydb = MySQLdb.connect(
    host="localhost",
    user="root",
    password="",
    database="portal"
)
mycursor = mydb.cursor()

width = 480
height = 320

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
    'identitas' : '',
}

def tangkap_wajah(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    if face_locations:
        face_encoding = face_recognition.face_encodings(frame)[0]
        return face_encoding
    else :
        return None

def face_detect(frame):
    fname = 'static/ref_name.pkl'
    with open(fname, 'rb') as fn:
        ref_dictt = pickle.load(fn)

    fembed = 'static/ref_embed.pkl'
    with open(fembed, 'rb') as fe:
        embed_dictt = pickle.load(fe)

    known_face_encodings = []
    known_face_names = []

    for ref_id, embed_list in embed_dictt.items():
        for my_embed in embed_list:
            known_face_encodings.append(my_embed)
            known_face_names.append(ref_id)

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        identitas = "N/A"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            ref_id = known_face_names[best_match_index]
            if ref_id in ref_dictt:
                detect_face['id_ref'] = ref_id
                name = ref_dictt[ref_id].get("name", "Unknown")
                identitas = ref_dictt[ref_id].get("identitas", "N/A")
        face_names.append((name, identitas))
        detect_face['nama'] = name
        detect_face['identitas'] = identitas

    for (top_s, right, bottom, left), (name, identitas) in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top_s), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"{name}", (left + 6, bottom - 36), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"{identitas}", (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

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

@app.route('/login', methods = ['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        sql = 'SELECT * FROM user WHERE username = %s'
        mycursor.execute(sql, (username,))
        user = mycursor.fetchone()
        if user and bcrypt.checkpw(password.encode('utf-8'), user[2].encode('utf-8')):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return '<script>alert("Username atau Password salah!"); window.location.href = "/";</script>'

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/ganti_password', methods = ['GET', 'POST'])
def ganti_password():
    if 'username' not in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            return '<script>alert("Password tidak sesuai"); window.location.href = "/ganti_password";</script>'

        username = session['username']
        hash_new_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        sql_update = 'UPDATE user SET password = %s WHERE username = %s'
        mycursor.execute(sql_update, (hash_new_password.decode('utf-8'), username))
        mydb.commit()
        return '<script>alert("Password telah diubah"); window.location.href = "/home";</script>'

    return render_template('ganti_password.html')

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('/'))
    sql = 'SELECT gambar_wajah, gambar_plat FROM log ORDER BY id DESC LIMIT 2'
    mycursor.execute(sql)
    result = mycursor.fetchall()
    image_paths = [(row[0], row[1]) for row in result]
    return render_template('home.html', image_paths=image_paths)

@app.route('/save', methods=['POST'])
def save():
    current_time = datetime.now()

    image_wajah = f'{detect_face["identitas"]}_{current_time.strftime("%Y_%m_%d_%H_%M_%S")}.png'
    image_wajah = image_wajah.replace(':', '').replace('_', '').replace(' ', '').replace('-', '')
    image_wajah_path = os.path.join('static/imagewajah', image_wajah)
    cv2.imwrite(image_wajah_path, detect_face['frame'])

    image_plat = f'{detect_plate["plate"]}_{current_time.strftime("%Y_%m_%d_%H_%M_%S")}.png'
    image_plat = image_plat.replace(':', '').replace('_', '').replace(' ', '').replace('-', '')
    image_plat_path = os.path.join('static/imageplat', image_plat)
    cv2.imwrite(image_plat_path, detect_plate['frame'])

    query = 'INSERT INTO log (id_profile, plat, gambar_wajah, gambar_plat) VALUES ("{}", "{}", "{}", "{}")'.format(
        detect_face['identitas'],
        detect_plate['plate'],
        image_wajah,
        image_plat
    )

    print(query)
    mycursor.execute(query)
    mydb.commit()
    img_wajah_url = f'/static/imagewajah/{image_wajah}'
    img_plat_url = f'/static/imageplat/{image_plat}'

    return jsonify({'message': True, 'imgWajahURL': img_wajah_url, 'imgPlatURL': img_plat_url})

@app.route('/pendaftaran', methods=['GET'])
def pendaftaran():
    sql = 'SELECT * FROM identitas'
    mycursor.execute(sql)
    identitas = [row for row in mycursor.fetchall()]

    return render_template('pendaftaran.html', identitas=identitas)

@app.route('/submit', methods=['POST'])
def submit():
    try:
        f = open("/portal/static/ref_name.pkl", "rb")
        ref_dict_update = pickle.load(f)
        f.close()
    except FileNotFoundError:
        ref_dict_update = {}

    # Auto-increment the ID
    if not ref_dict_update:
        ref_id = 1
    else:
        ref_id = max(ref_dict_update.keys()) + 1

    ref_dict_update[ref_id] = {"name": request.form['nama'], "identitas": request.form['no_identitas'] }

    f = open("/portal/static/ref_name.pkl", "wb")
    pickle.dump(ref_dict_update, f)
    f.close()

    try:
        f = open("/portal/static/ref_embed.pkl", "rb")
        embed_dict_update = pickle.load(f)
        f.close()
    except FileNotFoundError:
        embed_dict_update = {}

    embed_list = request.form['embed_dict']
    embed_list = json.loads(embed_list)
    new_emb = []
    for emb in embed_list:
        new_emb.append(np.array(emb[0]))
    print(type(new_emb[0]))
    embed_dict_update[ref_id] = new_emb

    f = open("/portal/static/ref_embed.pkl", "wb")
    pickle.dump(embed_dict_update, f)
    print(f)
    f.close()

    no_identitas = request.form.get('no_identitas')
    nama = request.form.get('nama')
    id_identitas = int(request.form.get('id_identitas'))

    if id_identitas == 1:
        sql2 = 'INSERT INTO profile (no_identitas, nama, id_identitas) VALUES (%s, %s, 1)'
        val = (no_identitas, nama)
        print(val)
    elif id_identitas == 2:
        sql2 = 'INSERT INTO profile (no_identitas, nama, id_identitas) VALUES (%s, %s, 2)'
        val = (no_identitas, nama)
        print(val)
    elif id_identitas == 3:
        sql2 = 'INSERT INTO profile (no_identitas, nama, id_identitas) VALUES (%s, %s, 3)'
        val = (no_identitas, nama)
        print(val)
    else:
        return jsonify({'message': False})

    mycursor.execute(sql2, val)
    mydb.commit()
    return jsonify({'message': True})

@app.route('/tangkap', methods=['POST'])
def tangkap():
    frame_data = request.form['frame']
    
    # Decode base64 encoded string to numpy array
    frame_data_decoded = base64.b64decode(frame_data.split(',')[1])
    nparr = np.frombuffer(frame_data_decoded, np.uint8)
    
    # Decode image
    frame_capture = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    small_frame = cv2.resize(frame_capture, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    if not face_locations:
        return jsonify({'msg': 'Wajah tidak terdeteksi'})
    face_encoding = face_recognition.face_encodings(frame_capture)[0]
    print(len(face_encoding))

    # Encode gambar kembali ke dalam base64 encoded string
    _, buffer = cv2.imencode('.jpg', frame_capture)
    frame_data_encoded = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({'msg': 'success', 'img': frame_data_encoded, 'face_encoding':face_encoding.tolist()})

@app.route('/history')
def history():
    sql = 'SELECT profile.nama, identitas.nama_identitas, log.* FROM log JOIN profile ON profile.no_identitas=log.id_profile JOIN identitas ON identitas.id=profile.id_identitas ORDER BY log.id DESC;'
    mycursor.execute(sql)
    container = [row for row in mycursor.fetchall()]

    return render_template('history.html', container=container)

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

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
