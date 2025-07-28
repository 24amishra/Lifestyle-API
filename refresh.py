 
import requests
import base64
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
from langchain_openai import ChatOpenAI




refresh_token = 'b0dfd88650286360849639229ddbef54d90733481207e8ba7517488d6598e9e3'
client_id = '23RTQM'
access_token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyM1JUUU0iLCJzdWIiOiJCWkhYUlQiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJ3aHIgd3BybyB3bnV0IHdzbGUgd2VjZyB3c29jIHdhY3Qgd294eSB3dGVtIHd3ZWkgd2NmIHdzZXQgd3JlcyB3bG9jIiwiZXhwIjoxNzQ5NjIzOTEwLCJpYXQiOjE3NDk1OTUxMTB9.YNso_1rWYQWc5DOl7JZxahbNvvmZPpEpIY8MFur7rmw'
data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id
    }


#{'access_token': 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyM1JUUU0iLCJzdWIiOiJCWkhYUlQiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJ3aHIgd3BybyB3bnV0IHdzbGUgd2VjZyB3c29jIHdhY3Qgd294eSB3dGVtIHd3ZWkgd2NmIHdzZXQgd3JlcyB3bG9jIiwiZXhwIjoxNzQ5ODc0MTMwLCJpYXQiOjE3NDk4NDUzMzB9.hCeCGCHP73-zVBsKJOsDHwgfyGPXdZlHS8kRz5mk3Oo', 
# 'expires_in': 28800, '
# refresh_token': 'b0dfd88650286360849639229ddbef54d90733481207e8ba7517488d6598e9e3'
access_token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyM1JUUU0iLCJzdWIiOiJCWkhYUlQiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJ3aHIgd251dCB3cHJvIHdzbGUgd2VjZyB3c29jIHdhY3Qgd294eSB3dGVtIHd3ZWkgd2NmIHdzZXQgd2xvYyB3cmVzIiwiZXhwIjoxNzQ4NTg1ODM1LCJpYXQiOjE3NDg1NTcwMzV9.nGnDiPwCZdBQ7q4vN8L6OgxFY40nbP9Zurb15vbihKE'
headers2 = {"Content-Type": "application/x-www-form-urlencoded"}
res = requests.post('https://api.fitbit.com/oauth2/token', data=data, headers=headers2)
res = res.json()
print(res)