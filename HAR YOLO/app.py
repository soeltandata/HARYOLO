from flask import Flask, render_template, Response, session

# FlaskForm--> dibutuhkan untuk menerima input dari pengguna

from flask_wtf import FlaskForm

from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os


import cv2 # dibutuhkan untuk menajalankan YOLOv8

# YOLO_Video is the python file which contains the code for our object detection model
# YOLO_Video adalah file python yang berisi kode untuk menjalankan fungsi web
# video_detection, image_detection, webcam_detection adalah Function yang akan dipanggil
from YOLO_Video import video_detection, image_detection, webcam_detection

app = Flask(__name__)

app.config['SECRET_KEY'] = 'sultan123'
app.config['UPLOAD_FOLDER'] = 'static/files'


# Use FlaskForm to get input video file  from user
class UploadFileForm(FlaskForm):
    # We store the uploaded video file path in the FileField in the variable file
    # We have added validators to make sure the user inputs the video in the valid format  and user does upload the
    # video when prompted to do so
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames_video(path_x=''): # untuk melakukan generate frames pada video yang telah di upload
    yolo_output = video_detection(path_x) # menggunakan video detection pada file yang telah di upload sesuai path
    for detection_ in yolo_output:  # looping untuk melakukan generate frames
        if detection_ is None:
            print("Generate Video is done")  # jika proses generate telah selesai, alert terminal
        else:
            ref, buffer = cv2.imencode('.jpg', detection_) # melakukan encoding yang wajib untuk aplikasi berbasis flask
            frame = buffer.tobytes() # mengubah frame menjadi bytes
            yield (b'--frame\r\n' # melakukan yield untuk menyatukan frame menjadi video
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def generate_frames_image(path_y): # untuk melakukan generate frames pada gambar yang telah di upload
    print("generate_frames_image" + path_y)
    yolo_output = image_detection(path_y)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_web(path_z): # untuk melakukan generate frames pada webcam
    yolo_output = webcam_detection(path_z)
    # proses encoding dan mengubah menjadi bytes serta menyatukan semua frame menjadi satu result
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET', 'POST']) # menggunakan method 'app.route()' , untuk render halaman home sebagai halaman utama pada "/"
@app.route('/home', methods=['GET', 'POST']) # menggunakan method 'app.route()' , untuk render halaman home pada "/home"
def home():
    session.clear() # membersihkan storage session
    return render_template('home.html') # render halaman home


@app.route('/GambarPage', methods=['GET', 'POST']) # menggunakan method 'app.route()' , untuk render halaman gambar  pada "/GambarPage"
def gambarpage():
    form = UploadFileForm() # Upload File Form: membuat objek untuk upload form pengguna
    if form.validate_on_submit():
        file = form.file.data # file gambar akan disimpan pada path di bawah ini
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # melakukan save
        # menggunakan session storage untuk menyimpan image path di bawah ini
        session['ImagePath'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('gambar.html', form=form) # render halaman gambar

@app.route("/WebcamPage", methods=['GET', 'POST']) # menggunakan method 'app.route()' , untuk render halaman webcam pada "/WebcamPage"
def webcampage():
    session.clear() # membersihkan storage session
    return render_template('webcam.html') # Render halaman webcam


@app.route('/VideoPage', methods=['GET', 'POST']) # menggunakan method 'app.route()' , untuk render halaman video pada "/VideoPage"
def videopage():
    # Upload File Form: membuat objek untuk upload form pengguna
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # file video akan disimpan pada path di bawah ini
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # melakukan save
        session['loading'] = True
        # menggunakan session storage untuk menyimpan video path di bawah ini
        session['VideoPath'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('video.html', form=form) # render halaman video


@app.route('/video')
def video():
    # memanggil fungsi generate_frames_video untuk memproses video memberikan output
    return Response(generate_frames_video(path_x=session.get('VideoPath', None)),
                    mimetype='multipart/x-mixed-replace; '
                             'boundary=frame')


@app.route('/gambar')
def gambar():
    # memanggil fungsi generate_frames_image untuk memproses gambar dan memberikan output
    return Response(generate_frames_image(path_y=session.get('ImagePath', None)),
                    mimetype='multipart/x-mixed-replace;boundary=frame')


# To display the Output Video on Webcam page
@app.route('/webapp')
def webapp():
    # memanggil fungsi generate_frames_web untuk memproses input webcam dan memberikan output
    return Response(generate_frames_web(path_z=0), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # melakukan run dan debug menggunakan flask
    app.run(debug=True)
