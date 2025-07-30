import requests
import base64
from flask import Flask, render_template, url_for, request, jsonify,session,Response,redirect
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
import json
import time

import datetime
from urllib.parse import quote
import secrets
import hashlib
import base64
from urllib.parse import quote, urlencode
from pymongo import MongoClient
import time


client = MongoClient(os.getenv("MONGO_URI"))
db = client['FitAuth']
collection = db['Details']


app = Flask(__name__)
app.secret_key = "Pluto@1234"
CORS(app, origins=[
    "https://lifestyle-api-2.onrender.com",  # Your frontend URL
    "http://localhost:3000"  # For local development
], supports_credentials=True)

def save_tokens(access_token, refresh_token, client_id="23QFHC", user_id=None):
    """
    Save tokens to MongoDB. If user_id is provided, update existing user.
    Otherwise, create new document.
    """
    print("Saving tokens to MongoDB")
    
    token_data = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "client_id": client_id,
        "updated_at": datetime.datetime.utcnow()
    }
    
    try:
        if user_id:
            # Update existing user's tokens
            result = collection.update_one(
                {"_id": user_id},
                {"$set": token_data}
            )
            if result.modified_count > 0:
                print(f"Updated tokens for user {user_id}")
                return user_id
            else:
                print(f"No user found with ID {user_id}, creating new document")
        
        # Create new document if no user_id provided or user not found
        token_data["created_at"] = datetime.datetime.utcnow()
        result = collection.insert_one(token_data)
        print(f"Created new token document with ID: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"Error saving tokens: {e}")
        return None

def load_tokens(user_id=None):
    """
    Load tokens from MongoDB. If user_id provided, load specific user's tokens.
    Otherwise, load the most recent tokens.
    """
    print("Loading tokens from MongoDB")
    
    try:
        if user_id:
            # Load specific user's tokens
            from bson import ObjectId
            if isinstance(user_id, str):
                user_id = ObjectId(user_id)
            
            document = collection.find_one({"_id": user_id})
        else:
            # Load most recent tokens (fallback for existing functionality)
            document = collection.find_one(sort=[("updated_at", -1)])
        
        if document:
            access_token = document.get("access_token")
            refresh_token = document.get("refresh_token")
            print(f"Loaded tokens for user: {document.get('_id')}")
            return access_token, refresh_token, str(document.get("_id"))
        else:
            print("No tokens found in database")
            return None, None, None
            
    except Exception as e:
        print(f"Error loading tokens: {e}")
        return None, None, None
def generate_code_challenge(code_verifier):
    """Generate code challenge from code verifier using SHA256 and base64url encoding"""
    # SHA-256 hash of the code verifier
    sha256_hash = hashlib.sha256(code_verifier.encode('utf-8')).digest()
    
    # Base64url encode and remove padding
    code_challenge = base64.urlsafe_b64encode(sha256_hash).decode('utf-8').rstrip('=')
    
    return code_challenge




import base64
import os
import requests

def refreshing_token(refresh_token, client_id, user_doc_id=None):
    client_secret = os.getenv("CLIENT_SECRET")
    client_id = '23QFHC'
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id
    }
    client_creds = f"{client_id}:{client_secret}"
    base64_creds = base64.b64encode(client_creds.encode()).decode()
    headers2 = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {base64_creds}",
    }
    
    print("REFRESHING TOKEN")
    res = requests.post('https://api.fitbit.com/oauth2/token', data=data, headers=headers2)
    
    if res.status_code == 200:
        token_data = res.json()
        print("Token refresh successful")
        
        # Update tokens in MongoDB
        if user_doc_id:
            save_tokens(
                token_data["access_token"],
                token_data["refresh_token"],
                client_id,
                user_doc_id
            )
        
        return token_data
    else:
        print("Token refresh failed", res.status_code)
        print(res.text)
        return None
    

def workflow():
    pass
def code_verif():
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')


@app.route('/callback')
def callback():
    code = request.args.get('code')
    print(code)
    print(session.get('code_verifier'))

    client_id = os.getenv("FITBIT_CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    redirect_uri = os.getenv("REDIRECT_URI")

    token_url = "https://api.fitbit.com/oauth2/token"
    headers = {
        "Authorization": "Basic " + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode(),
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "client_id": client_id,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
        "code": code,
        "code_verifier": session.get('code_verifier'),
    }

    res = requests.post(token_url, headers=headers, data=data)
    if res.status_code == 200:
        tokens = res.json()
        
        # Save tokens to MongoDB and get the document ID
        user_doc_id = save_tokens(
            tokens["access_token"], 
            tokens["refresh_token"],
            client_id
        )
        
        # Store the user document ID in session for this user
        session['user_doc_id'] = user_doc_id
        
        return "Authorized! You can now access your Fitbit data."
    else:
        return f"Error: {res.status_code} {res.text}"



@app.route('/authorize')
def authorize():
    code_verifier = code_verif()
    code_challenge = generate_code_challenge(code_verifier)
    session['code_verifier'] = code_verifier
    session['code_challenge'] = code_challenge  # You don't actually need to store this
    
    client_id = os.getenv("FITBIT_CLIENT_ID")
    redirect_uri = "http://localhost:8000/callback"
    scope = "activity heartrate sleep profile"
    
    # Use urlencode to properly format the URL
    params = {
        'response_type': 'code',
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'scope': scope,
        'code_challenge': code_challenge,
        'code_challenge_method': 'S256'
    }
    
    auth_url = f"https://www.fitbit.com/oauth2/authorize?{urlencode(params)}"
    return redirect(auth_url)
    

@app.route('/endpoint', methods=['POST','GET'])
def chatbot():
    load_dotenv()  
    openai_key = os.getenv("OPENAI_API_KEY")

    # Get user_doc_id from session (set during OAuth callback)
    user_doc_id = session.get('user_doc_id')
    
    # Load tokens from MongoDB
    access_token, refresh_token, doc_id = load_tokens(user_doc_id)
    
    if not access_token or not refresh_token:
        return jsonify({'error': 'No valid tokens found. Please authorize first.'}), 401

    client_id = '23QFHC'
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    # Set up headers with loaded access token
    headers = {
        'Authorization': f'Bearer {access_token}',
    } 
    
    # API endpoints
    activityUrl = f'https://api.fitbit.com/1/user/-/activities/steps/date/{today}/30d.json'
    sleepUrl = f"https://api.fitbit.com/1/user/-/sleep/date/{today}.json"
    heartUrl = f'https://api.fitbit.com/1/user/-/activities/heart/date/{today}/30d.json'
    calUrl = f"https://api.fitbit.com/1/user/-/activities/calories/date/{today}/30d.json"

    # Initialize response variables
    stepRes = ""
    sleepRes = ""
    heartRes = ""
    calRes = ""

    def make_api_calls_with_refresh(headers, max_retries=1):
        """Make API calls with automatic token refresh if needed"""
        nonlocal access_token, refresh_token
        
        for attempt in range(max_retries + 1):
            print(f"API call attempt {attempt + 1}")
            
            # Make all API calls
            response = requests.get(activityUrl, headers=headers)
            sleepResponse = requests.get(sleepUrl, headers=headers)
            heartResponse = requests.get(heartUrl, headers=headers)
            calResponse = requests.get(calUrl, headers=headers)
            
            # Check if any call failed due to token issues
            failed_calls = []
            if response.status_code == 401:
                failed_calls.append("steps")
            if sleepResponse.status_code == 401:
                failed_calls.append("sleep")
            if heartResponse.status_code == 401:
                failed_calls.append("heart")
            if calResponse.status_code == 401:
                failed_calls.append("calories")
            
            # If no 401 errors, process the responses
            if not failed_calls:
                results = {
                    'steps': None,
                    'sleep': None, 
                    'heart': None,
                    'calories': None
                }
                
                # Process Steps Data
                if response.status_code == 200:
                    resJson = response.json()
                    results['steps'] = resJson['activities-steps']
                    print("✓ Steps data retrieved")
                else:
                    print(f"✗ Steps data failed: {response.status_code}")

                # Process Sleep Data
                if sleepResponse.status_code == 200:
                    sleepJson = sleepResponse.json()
                    results['sleep'] = sleepJson
                    print("✓ Sleep data retrieved")
                else:
                    print(f"✗ Sleep data failed: {sleepResponse.status_code}")

                # Process Heart Rate Data
                if heartResponse.status_code == 200:
                    heartJson = heartResponse.json()
                    results['heart'] = heartJson['activities-heart']
                    print("✓ Heart rate data retrieved")
                else:
                    print(f"✗ Heart rate data failed: {heartResponse.status_code}")

                # Process Calories Data
                if calResponse.status_code == 200:
                    calJson = calResponse.json()
                    results['calories'] = calJson['activities-calories']
                    print("✓ Calories data retrieved")
                else:
                    print(f"✗ Calories data failed: {calResponse.status_code}")
                
                return results
            
            # If we have 401 errors and haven't exhausted retries, refresh token
            if attempt < max_retries:
                print(f"Token expired for: {', '.join(failed_calls)}. Refreshing token...")
                refreshed_tokens = refreshing_token(refresh_token, client_id, doc_id)
                
                if refreshed_tokens:
                    access_token = refreshed_tokens['access_token']
                    refresh_token = refreshed_tokens['refresh_token']
                    
                    # Update headers with new token
                    headers["Authorization"] = f"Bearer {access_token}"
                    print("Token refreshed successfully, retrying API calls...")
                else:
                    print("Token refresh failed")
                    return None
            else:
                print("Max retries exceeded")
                return None
        
        return None

    # Make API calls with automatic token refresh
    start_time = time.time()
    api_results = make_api_calls_with_refresh(headers)
    end_time = time.time()
    print(f"API calls completed in {end_time - start_time:.2f} seconds")
    
    if not api_results:
        return jsonify({'error': 'Failed to retrieve Fitbit data. Please re-authorize.'}), 401
    
    # Extract results
    stepRes = api_results['steps']
    sleepRes = api_results['sleep']
    heartRes = api_results['heart']
    calRes = api_results['calories']

    # Combine all data into one comprehensive string for LLM
    def format_fitbit_data_for_llm(step_data, sleep_data, heart_data, cal_data, date):
        combined_data = f"""
COMPREHENSIVE FITBIT HEALTH DATA SUMMARY for {date}

=== 30-DAY STEPS HISTORY ===
"""
        
        # Format Steps Data
        if step_data:
            combined_data += "Daily Steps for the last 30 days:\n"
            for day in step_data:
                combined_data += f"  {day['dateTime']}: {day['value']} steps\n"
            
            # Calculate averages
            recent_steps = [int(day['value']) for day in step_data[-7:]]
            week_avg = sum(recent_steps) / len(recent_steps) if recent_steps else 0
            month_avg = sum(int(day['value']) for day in step_data) / len(step_data) if step_data else 0
            combined_data += f"\n7-day average: {week_avg:.0f} steps\n"
            combined_data += f"30-day average: {month_avg:.0f} steps\n"
        else:
            combined_data += "Steps data not available\n"
        
        combined_data += "\n=== 30-DAY CALORIES HISTORY ===\n"
        
        # Format Calories Data
        if cal_data:
            combined_data += "Daily Calories Burned for the last 30 days:\n"
            for day in cal_data:
                combined_data += f"  {day['dateTime']}: {day['value']} calories\n"
            
            # Calculate averages
            recent_cals = [int(day['value']) for day in cal_data[-7:]]
            week_avg_cal = sum(recent_cals) / len(recent_cals) if recent_cals else 0
            month_avg_cal = sum(int(day['value']) for day in cal_data) / len(cal_data) if cal_data else 0
            combined_data += f"\n7-day average: {week_avg_cal:.0f} calories\n"
            combined_data += f"30-day average: {month_avg_cal:.0f} calories\n"
        else:
            combined_data += "Calories data not available\n"
        
        combined_data += "\n=== 30-DAY HEART RATE HISTORY ===\n"
        
        # Format Heart Rate Data
        if heart_data:
            combined_data += "Daily Heart Rate Data for the last 30 days:\n"
            for day in heart_data:
                resting_hr = day.get('value', {}).get('restingHeartRate', 'N/A')
                combined_data += f"  {day['dateTime']}: Resting HR = {resting_hr} bpm\n"
            
            # Calculate average resting heart rate
            resting_hrs = [day.get('value', {}).get('restingHeartRate', 0) 
                          for day in heart_data 
                          if day.get('value', {}).get('restingHeartRate')]
            if resting_hrs:
                avg_resting_hr = sum(resting_hrs) / len(resting_hrs)
                combined_data += f"\nAverage resting heart rate: {avg_resting_hr:.0f} bpm\n"
        else:
            combined_data += "Heart rate data not available\n"
        
        combined_data += "\n=== TODAY'S SLEEP SUMMARY ===\n"
        
        # Format Sleep Data
        if sleep_data and sleep_data.get('sleep'):
            sleep_record = sleep_data['sleep'][0] if sleep_data['sleep'] else {}
            
            combined_data += f"Sleep data for {date}:\n"
            combined_data += f"  Total sleep time: {sleep_record.get('minutesAsleep', 0)} minutes\n"
            combined_data += f"  Time in bed: {sleep_record.get('timeInBed', 0)} minutes\n"
            combined_data += f"  Sleep efficiency: {sleep_record.get('efficiency', 0)}%\n"
            combined_data += f"  Times awakened: {sleep_record.get('awakensCount', 0)}\n"
            combined_data += f"  Minutes awake: {sleep_record.get('minutesAwake', 0)}\n"
            
            # Sleep stages if available
            levels = sleep_record.get('levels', {})
            if levels and levels.get('summary'):
                summary_levels = levels['summary']
                combined_data += "\nSleep Stages:\n"
                combined_data += f"  Deep sleep: {summary_levels.get('deep', {}).get('minutes', 0)} minutes\n"
                combined_data += f"  Light sleep: {summary_levels.get('light', {}).get('minutes', 0)} minutes\n"
                combined_data += f"  REM sleep: {summary_levels.get('rem', {}).get('minutes', 0)} minutes\n"
                combined_data += f"  Wake time: {summary_levels.get('wake', {}).get('minutes', 0)} minutes\n"
        else:
            combined_data += "Sleep data not available for today\n"
        
        combined_data += "\n=== ANALYSIS NOTES ===\n"
        combined_data += "This data represents comprehensive health metrics for a cancer survivor.\n"
        combined_data += "Please provide realistic, safe recommendations based on this activity pattern.\n"
        
        return combined_data

    # Create the combined string
    combined_fitbit_data = format_fitbit_data_for_llm(stepRes, sleepRes, heartRes, calRes, today)
    
    print("Combined data string created for LLM")
    print(f"Data string length: {len(combined_fitbit_data)} characters")

    # Continue with LLM processing
    start_time = time.time()
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=openai_key
    )

    messages = [
        (
            "system",
            "You are a helpful assistant supporting a cancer survivor. Based on this comprehensive Fitbit data, suggest a realistic step goal for the next 7 days. Consider their recent patterns, sleep quality, and overall activity trends. Keep recommendations safe and achievable."
        ),
        (
            "human", 
            combined_fitbit_data
        )
    ]
    
    ai_msg = llm.invoke(messages)
    end_time = time.time()
    print(f"LLM processing time: {end_time - start_time:.2f} seconds")
    
    return jsonify({
        'evaluation': ai_msg.content,
        'raw_data': {
            'steps': stepRes,
            'sleep': sleepRes,
            'heart': heartRes,
            'calories': calRes
        },
        'summary': combined_fitbit_data
    })
       

if __name__ == '__main__':

    app.run(port=8000, debug=True)
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

