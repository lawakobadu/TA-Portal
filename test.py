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

@app.route('/')
def index():
    return 'Hello World'

app.run(debug=True)