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
import gc


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
    redirect_uri = "https://lifestyle-api.onrender.com/callback"
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
    """Optimized version with memory management"""
    
    # Clear any existing memory
    gc.collect()
    
    load_dotenv()  
    openai_key = os.getenv("OPENAI_API_KEY")

    # Get user_doc_id from session
    user_doc_id = session.get('user_doc_id')
    
    # Load tokens - add timeout
    try:
        access_token, refresh_token, doc_id = load_tokens(user_doc_id)
    except Exception as e:
        print(f"Token loading failed: {e}")
        return jsonify({'error': 'Database connection timeout'}), 500
    
    if not access_token or not refresh_token:
        return jsonify({'error': 'No valid tokens found. Please authorize first.'}), 401

    client_id = '23QFHC'
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    headers = {'Authorization': f'Bearer {access_token}'}
    
    # Simplified API URLs - get less data to save memory
    activityUrl = f'https://api.fitbit.com/1/user/-/activities/steps/date/{today}/7d.json'  # 7 days instead of 30
    sleepUrl = f"https://api.fitbit.com/1/user/-/sleep/date/{today}.json"
    heartUrl = f'https://api.fitbit.com/1/user/-/activities/heart/date/{today}/7d.json'   # 7 days instead of 30
    calUrl = f"https://api.fitbit.com/1/user/-/activities/calories/date/{today}/7d.json"  # 7 days instead of 30

    def make_api_calls_optimized(headers, max_retries=1):
        """Memory-optimized API calls"""
        nonlocal access_token, refresh_token
        
        for attempt in range(max_retries + 1):
            print(f"API call attempt {attempt + 1}")
            
            try:
                # Make API calls with shorter timeout
                response = requests.get(activityUrl, headers=headers, timeout=10)
                sleepResponse = requests.get(sleepUrl, headers=headers, timeout=10)
                heartResponse = requests.get(heartUrl, headers=headers, timeout=10)
                calResponse = requests.get(calUrl, headers=headers, timeout=10)
                
                # Check for 401 errors
                failed_calls = []
                if response.status_code == 401:
                    failed_calls.append("steps")
                if sleepResponse.status_code == 401:
                    failed_calls.append("sleep")
                if heartResponse.status_code == 401:
                    failed_calls.append("heart")
                if calResponse.status_code == 401:
                    failed_calls.append("calories")
                
                # If no 401 errors, process responses
                if not failed_calls:
                    results = {'steps': None, 'sleep': None, 'heart': None, 'calories': None}
                    
                    # Process each response and immediately free memory
                    if response.status_code == 200:
                        results['steps'] = response.json()['activities-steps']
                        print("✓ Steps data retrieved")
                    del response  # Free memory immediately
                    
                    if sleepResponse.status_code == 200:
                        results['sleep'] = sleepResponse.json()
                        print("✓ Sleep data retrieved")
                    del sleepResponse  # Free memory immediately
                    
                    if heartResponse.status_code == 200:
                        results['heart'] = heartResponse.json()['activities-heart']
                        print("✓ Heart rate data retrieved")
                    del heartResponse  # Free memory immediately
                    
                    if calResponse.status_code == 200:
                        results['calories'] = calResponse.json()['activities-calories']
                        print("✓ Calories data retrieved")
                    del calResponse  # Free memory immediately
                    
                    # Force garbage collection
                    gc.collect()
                    return results
                
                # Handle token refresh if needed
                if attempt < max_retries and failed_calls:
                    print(f"Token expired for: {', '.join(failed_calls)}. Refreshing...")
                    refreshed_tokens = refreshing_token(refresh_token, client_id, doc_id)
                    
                    if refreshed_tokens:
                        access_token = refreshed_tokens['access_token']
                        refresh_token = refreshed_tokens['refresh_token']
                        headers["Authorization"] = f"Bearer {access_token}"
                        print("Token refreshed, retrying...")
                    else:
                        return None
                else:
                    return None
                    
            except requests.exceptions.Timeout:
                print("API request timeout")
                return None
            except Exception as e:
                print(f"API request error: {e}")
                return None
        
        return None

    # Make API calls
    api_results = make_api_calls_optimized(headers)
    
    if not api_results:
        return jsonify({'error': 'Failed to retrieve Fitbit data'}), 401
    
    # Create simplified data string (shorter to save memory)
    def format_fitbit_data_simple(step_data, sleep_data, heart_data, cal_data, date):
        """Simplified formatting to reduce memory usage"""
        combined_data = f"FITBIT HEALTH DATA for {date}\n\n"
        
        # Last 7 days only (instead of 30)
        if step_data:
            recent_steps = step_data[-7:]  # Last 7 days
            combined_data += "Recent Steps (7 days):\n"
            for day in recent_steps:
                combined_data += f"  {day['dateTime']}: {day['value']} steps\n"
            
            avg_steps = sum(int(day['value']) for day in recent_steps) / len(recent_steps)
            combined_data += f"Average: {avg_steps:.0f} steps\n\n"
        
        # Simplified sleep data
        if sleep_data and sleep_data.get('sleep'):
            sleep_record = sleep_data['sleep'][0] if sleep_data['sleep'] else {}
            combined_data += f"Sleep ({date}):\n"
            combined_data += f"  Total: {sleep_record.get('minutesAsleep', 0)} min\n"
            combined_data += f"  Efficiency: {sleep_record.get('efficiency', 0)}%\n\n"
        
        combined_data += "Provide brief, safe health recommendations for a cancer survivor.\n"
        return combined_data

    # Create simplified data string
    combined_fitbit_data = format_fitbit_data_simple(
        api_results['steps'], api_results['sleep'], 
        api_results['heart'], api_results['calories'], today
    )
    
    # Clear API results from memory
    del api_results
    gc.collect()
    
    print(f"Data string length: {len(combined_fitbit_data)} characters")

    # Use lighter LLM model and shorter context
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Lighter model
            temperature=0,
            max_tokens=500,       # Limit response length
            timeout=30,           # Shorter timeout
            max_retries=1,        # Fewer retries
            api_key=openai_key
        )

        messages = [
            ("system", "You are a helpful assistant. Provide brief, safe recommendations based on Fitbit data for a cancer survivor. Keep response under 300 words."),
            ("human", combined_fitbit_data)
        ]
        
        ai_msg = llm.invoke(messages)
        
        # Clean up
        del llm, messages, combined_fitbit_data
        gc.collect()
        
        return jsonify({
            'evaluation': ai_msg.content,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"LLM processing error: {e}")
        return jsonify({'error': 'AI processing failed', 'details': str(e)}), 500

if __name__ == '__main__':

    app.run(port=8000, debug=True)
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

