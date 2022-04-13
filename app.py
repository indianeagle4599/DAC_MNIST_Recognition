import flask
from flask import Flask, render_template, url_for, request
import pickle
import base64
import numpy as np
import cv2
import tensorflow as tf

# Initialize the useless part of the base64 encoded image.
init_Base64 = 21;

model = tf.keras.models.load_model('NN_model.h5')
key_dict = {'ज्ञ (gyan)': 0,
            'ट (tamaatar)': 1,
            'ठ (thos)': 2,
            'ड (damroo)': 3,
            'ढ (dhaal)': 4,
            'ण (ghoshna)': 5,
            'त (tabala)': 6,
            'थ (thaali)': 7,
            'द (dard)': 8,
            'ध (dhanush)': 9,
            'क (kavach)': 10,
            'न (namaste)': 11,
            'प (paagal)': 12,
            'फ (phool)': 13,
            'ब (bandar)': 14,
            'भ (bhaaloo)': 15,
            'character_25_ma': 16,
            'character_26_yaw': 17,
            'character_27_ra': 18,
            'character_28_la': 19,
            'character_29_waw': 20,
            'character_2_kha': 21,
            'character_30_motosaw': 22,
            'character_31_petchiryakha': 23,
            'character_32_patalosaw': 24,
            'character_33_ha': 25,
            'character_34_chhya': 26,
            'character_35_tra': 27,
            'character_36_gya': 28,
            'character_3_ga': 29,
            'character_4_gha': 30,
            'character_5_kna': 31,
            'character_6_cha': 32,
            'character_7_chha': 33,
            'character_8_ja': 34,
            'character_9_jha': 35,
            'digit_0': 36,
            'digit_1': 37,
            'digit_2': 38,
            'digit_3': 39,
            'digit_4': 40,
            'digit_5': 41,
            'digit_6': 42,
            'digit_7': 43,
            'digit_8': 44,
            'digit_9': 45}


def get_key(dictionary, val):
    for key, value in dictionary.items():
        if val == value:
            return key
    return None

# Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')

# First route : Render the initial drawing template
@app.route('/')
def home():
	return render_template('draw.html')


# Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():

        if request.method == 'POST':
                final_pred = None
                # Preprocess the image : set the image to 28x28 shape
                # Access the image
                draw = request.form['url']
                # Removing the useless part of the url.
                draw = draw[init_Base64:]
                # Decoding
                draw_decoded = base64.b64decode(draw)
                image = np.asarray(bytearray(draw_decoded), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
                # Resizing and reshaping to keep the ratio.
                newimage = np.asarray(image, np.uint8)
                # Resizing and reshaping to keep the ratio.
                resized = cv2.resize(newimage, (32, 32), interpolation=cv2.INTER_AREA)

                vect = np.asarray(resized, dtype="float32")/255
                vect = (vect.reshape(1, 32, 32, 1).astype('float32'))

                # Launch prediction
                my_prediction = list(model.predict(vect)[0])
                # Associating the index and its value within the dictionary
                final_pred = get_key(key_dict, my_prediction.index(max(my_prediction)))
                print(final_pred)

        return render_template('results.html', prediction =final_pred)


if __name__ == '__main__':
	app.run(debug=True)
