from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prakriti')
def prakriti():
    return "<h1>Prakriti Assessment Page</h1>"

@app.route('/disease')
def disease():
    return "<h1>Disease Detection Page</h1>"

@app.route('/wellness')
def wellness():
    return "<h1>General Wellness Page</h1>"

@app.route('/profile')
def profile():
    return "<h1>User Profile Page</h1>"

if __name__ == '__main__':
    app.run(debug = True)
