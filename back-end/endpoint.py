from flask import Flask, render_template, url_for, request, jsonify,session,Response
from flask_cors import CORS
import requests
import getpass
import os
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import load_dotenv
from openai import OpenAI






app = Flask(__name__)

CORS(app, origins=["http://localhost:3000"], supports_credentials=True,)

@app.route('/endpoint',methods = ['GET'])
def send_data():
    #data = request.get_json()
    intro = 'hello world'
    response = jsonify({'message':intro,})
    response.headers["Content-Type"] = "application/json"
    return response,200



@app.route('/fitbit',methods = ['GET','POST'])
def access():
    
    refresh_token = '91fee71a5748b8b73e1a23c402f94513c2c71119da78a5dc7b6cdac779efb26a'

    client_id = '23QHVG'


    data = {
    "grant_type": "refresh_token",
    "refresh_token": refresh_token,
    "client_id": client_id
}

    headers = {
    "Content-Type": "application/x-www-form-urlencoded"
    }

    res = requests.post('https://api.fitbit.com/oauth2/token', data=data, headers=headers)

    tokens = res.json()
    return jsonify({
        'tokens': tokens

    })

    


@app.route('/test',methods = ['GET'])
def chat():
    load_dotenv()  
    openai_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_key,temperature=0)

    response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "developer",
            "content": "Talk like a pirate."
        },
        {
            "role": "user",
            "content": "Are semicolons optional in JavaScript?"
        }
    ]
)




if __name__ == '__main__':

    app.run(port=8001, debug=True)




