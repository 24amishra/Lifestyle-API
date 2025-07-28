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




load_dotenv()  
openai_key = os.getenv("OPENAI_API_KEY")









{
  "user_id": "123ABC456",
  "profile": {
    "name": "Alex Morgan",
    "age": 29,
    "gender": "female",
    "location": "San Francisco, CA",
    "temperature_settings": "Fahrenheit",
    "timezone": "America/Los_Angeles",
    "member_since": "2019-08-01",
    "locale": "en_US"
  },
  "activity": {
    "steps": 10234,
    "distance_km": 7.8,
    "floors": 12,
    "active_minutes": 65,
    "calories_out": 2320
  },
  "weight": {
    "weight_kg": 62.3,
    "bmi": 22.4,
    "date": "2025-05-20"
  },
  "heartrate": {
    "resting_heart_rate": 61,
    "average_heart_rate": 74,
    "max_heart_rate": 142
  },
  "cardio_fitness": {
    "vo2_max": 38.2,
    "fitness_score": "Good"
  },
  "sleep": {
    "total_sleep_minutes": 437,
    "deep_minutes": 90,
    "light_minutes": 280,
    "rem_minutes": 67,
    "wake_minutes": 30
  },
  "respiratory_rate": {
    "avg_breaths_per_min": 16.2,
    "min": 14.5,
    "max": 18.3
  },
  "oxygen_saturation": {
    "average_spo2_percent": 97.2,
    "min_spo2_percent": 95.1,
    "date": "2025-05-19"
  },
  "electrocardiogram": {
    "ecg_reading": "Normal sinus rhythm",
    "last_recorded": "2025-05-18T14:25:00"
  },
  "nutrition": {
    "calories_in": 1850,
    "protein_g": 90,
    "carbs_g": 210,
    "fat_g": 55,
    "water_ml": 2000
  }
}

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
   api_key  = openai_key
)

messages = [
    (
        "system",
        "You are a helpful assistant in charge of helping a cancer patient. Either raise or lower their step estimation based on this data.",
    ),
    ("human", 
    
     """
     Here is the health data
    
  "activity": {
    "steps": 22000,
    "distance_km": 7.8,
    "floors": 12,
    "active_minutes": 65,
    "calories_out": 2320
  },
  "weight": {
    "weight_kg": 62.3,
    "bmi": 22.4,
    "date": "2025-05-20"
  },
  "heartrate": {
    "resting_heart_rate": 61,
    "average_heart_rate": 74,
    "max_heart_rate": 142
  },
  "cardio_fitness": {
    "vo2_max": 38.2,
    "fitness_score": "Good"
  },
    
    "
   
   
   
   
   
   
   
   
   
   
   
   Their step goal is 20000. Should we raise or lower the estimation? provide a new threshold.
   
   
   
   """
    ),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)