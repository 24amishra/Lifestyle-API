from flask import Flask, render_template, url_for, request, jsonify,session,Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"], supports_credentials=True,)

@app.route('/endpoint',methods = ['GET'])
def send_data():
    #data = request.get_json()
    intro = 'hello world'
    response = jsonify({'message':intro,})
    response.headers["Content-Type"] = "application/json"
    return response,200




    



if __name__ == '__main__':

    app.run(port=8001, debug=True)




