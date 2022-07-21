from flask import Flask
app = Flask(__name__)

@app.route('/hello',methods=['GET', 'POST'])
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=6006)