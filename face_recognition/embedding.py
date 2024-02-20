import cv2
import face_recognition
import pickle

name = input("Masukkan nama: ")
nim = input("Masukkan NIM: ")

# try:
#     f = open("/portal/static/ref_name.pkl", "rb")
#     ref_dict = pickle.load(f)
#     f.close()
# except FileNotFoundError:
ref_dict = {}

# Auto-increment the ID
if not ref_dict:
    ref_id = 1
else:
    ref_id = max(ref_dict.keys()) + 1

ref_dict[ref_id] = {"name": name, "nim": nim}  # Menyimpan nama dan NIM

f = open("/portal/static/ref_name.pkl", "wb")
pickle.dump(ref_dict, f)
f.close()

# try:
#     f = open("/portal/static/ref_embed.pkl", "rb")
#     embed_dict = pickle.load(f)
#     f.close()
# except FileNotFoundError:
embed_dict = {}

for i in range(5):
    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        check, frame = webcam.read()

        cv2.imshow("Capturing", frame)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        key = cv2.waitKey(1)

        if key == ord('s'):
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if face_locations:
                face_encoding = face_recognition.face_encodings(frame)[0]
                if ref_id in embed_dict:
                    embed_dict[ref_id] += [face_encoding]
                else:
                    embed_dict[ref_id] = [face_encoding]
                webcam.release()
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

f = open("/portal/static/ref_embed.pkl", "wb")
pickle.dump(embed_dict, f)
f.close()
