from flask import Flask, render_template, request
import os
from inference.count_capuchin_calls import getCount

recording_dir = r'app/input_recordings'

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods = ['POST'])  
def success():  
    if request.method == 'POST':
        f = request.files['audio_file']
        os.makedirs(recording_dir, exist_ok=True)
        f.save(fr'{recording_dir}/{f.filename}') 
        count = getCount(fr'{recording_dir}/{f.filename}')
        os.remove(fr'{recording_dir}/{f.filename}')
        return render_template('index.html', count=count)

if __name__ == "__main__":
    app.run(debug=False, port=8085, host='192.168.1.60')