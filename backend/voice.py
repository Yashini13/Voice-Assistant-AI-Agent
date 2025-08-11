import pandas as pd
import os
import speech_recognition as sr
from dotenv import load_dotenv
import pyttsx3
import json

# IMPORTANT: Install required packages if not already installed:
# pip install pandas openpyxl speechrecognition pyttsx3 pyaudio openai
load_dotenv() 
api_key_openai = os.getenv('OPENAI_API_KEY')
sheet_url = os.getenv('SHEET_URL')

# Handle different OpenAI library versions
try:
    from openai import OpenAI
    # New OpenAI library (v1.0+)
    client = OpenAI(api_key=api_key_openai)
    NEW_OPENAI = True
except ImportError:
    # Old OpenAI library
    import openai
    openai.api_key = api_key_openai
    NEW_OPENAI = False

# --- API Config ---
MODEL = 'gpt-4o-mini'  # Using GPT-4o-mini (most cost-effective option)

# --- Helper function for OpenAI API calls ---
def call_openai_api(messages, temperature=0, max_tokens=200):
    """Universal function to handle both old and new OpenAI library versions"""
    try:
        if NEW_OPENAI:
            # New OpenAI library (v1.0+)
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        else:
            # Old OpenAI library
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

# --- Load Property Database ---
def load_property_database():
    """Load and return the property database from CSV"""
    try:
        df = pd.read_csv('propdata.csv')
        print(f"✅ Loaded {len(df)} properties from database")
        return df
    except FileNotFoundError:
        print("❌ Error: propdata.csv not found!")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return pd.DataFrame()

# --- Load Broker Database ---
def load_broker_database():
    """Load and return the broker database from CSV"""
    try:
        df = pd.read_csv('brokdata.csv')
        print(f"✅ Loaded {len(df)} brokers from database")
        return df
    except FileNotFoundError:
        print("❌ Error: brokdata.csv not found!")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading broker database: {e}")
        return pd.DataFrame()

# Load the databases at startup
property_db = load_property_database()
broker_db = load_broker_database()

# --- Questions for Real Estate Form ---
questions = [
    ("Name", "What is your full name?"),
    ("Contact", "Please say your contact number."),
    ("RentOrBuy", "Are you looking to rent or buy a property?"),
    ("Parking", "Do you need parking space?"),
    ("Location", "Which location are you looking for the property in?"),
    ("Budget", "What is your budget?")
]

valid_answers = {}

# --- Text to Speech (TTS) Setup ---
engine = pyttsx3.init()
def speak(text):
    print(f"\nAgent Shreyash: {text}")
    engine.say(text)
    engine.runAndWait()

# --- Speech to Text (STT) Setup ---
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        # Increased timeout and phrase_time_limit for better recognition
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        try:
            response = recognizer.recognize_google(audio)
            print(f"You: {response}")
            return response
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that. Could you please repeat?")
            return listen()
        except sr.RequestError:
            speak("Speech recognition service is down.")
            return ""
        except sr.WaitTimeoutError:
            speak("I didn't hear anything. Could you please speak?")
            return listen()

# --- Check if query is real estate related ---
def is_real_estate_related(user_question):
    """Check if the user's question is related to real estate"""
    real_estate_keywords = [
        'property', 'properties', 'house', 'home', 'apartment', 'flat', 'villa', 
        'rent', 'buy', 'purchase', 'sale', 'location', 'budget', 'bedroom', 
        'bathroom', 'sqft', 'area', 'parking', 'price', 'cost', 'building', 
        'tower', 'complex', 'residential', 'commercial', 'lease', 'mortgage',
        'investment', 'real estate', 'broker', 'agent', 'listing', 'available'
    ]
    
    # Also check if locations from our database are mentioned
    if not property_db.empty:
        locations = property_db['Location'].dropna().str.lower().tolist()
        real_estate_keywords.extend(locations)
    
    user_question_lower = user_question.lower()
    return any(keyword in user_question_lower for keyword in real_estate_keywords)

# --- Find matching broker for location/building ---
def find_matching_broker(location_query):
    """Find a broker that matches the user's location preference"""
    if broker_db.empty:
        return None
    
    # Try to match location with broker's location
    location_lower = location_query.lower()
    
    # First try exact location match
    for _, broker in broker_db.iterrows():
        if pd.notna(broker['Location']) and location_lower in broker['Location'].lower():
            return broker
    
    # If no exact match, try partial match
    for _, broker in broker_db.iterrows():
        if pd.notna(broker['Location']):
            broker_location_words = broker['Location'].lower().split()
            location_words = location_lower.split()
            if any(word in broker_location_words for word in location_words):
                return broker
    
    # If still no match, return first available broker
    return broker_db.iloc[0] if len(broker_db) > 0 else None

# --- Search Properties in Database ---
def search_properties(query_params):
    """Search properties based on user criteria"""
    if property_db.empty:
        return []
    
    filtered_df = property_db.copy()
    
    # Filter by location if specified
    if 'location' in query_params and query_params['location']:
        location_filter = filtered_df['Location'].str.contains(
            query_params['location'], case=False, na=False
        )
        filtered_df = filtered_df[location_filter]
    
    # Filter by property type if specified (using correct column name)
    if 'property_type' in query_params and query_params['property_type']:
        type_filter = filtered_df['Property Type'].str.contains(
            query_params['property_type'], case=False, na=False
        )
        filtered_df = filtered_df[type_filter]
    
    # Filter by budget if specified (using Purchase Price USD)
    if 'max_budget' in query_params and query_params['max_budget']:
        try:
            budget = float(query_params['max_budget'])
            # Check both purchase price and monthly rent
            price_filter = (filtered_df['Purchase Price (USD)'] <= budget) | (filtered_df['Monthly Rent (USD)'] <= budget)
            filtered_df = filtered_df[price_filter]
        except:
            pass
    
    # Filter by parking if specified
    if 'has_parking' in query_params and query_params['has_parking'] is not None:
        if query_params['has_parking']:
            filtered_df = filtered_df[filtered_df['Parking Spots'] > 0]
        else:
            filtered_df = filtered_df[filtered_df['Parking Spots'] == 0]
    
    return filtered_df.head(5).to_dict('records')  # Return top 5 matches

# --- Extract Search Parameters from User Query ---
def extract_search_params(user_question):
    """Use AI to extract search parameters from user question"""
    # Get available locations and property types from database (using correct column names)
    locations = property_db['Location'].dropna().unique().tolist() if not property_db.empty else []
    property_types = property_db['Property Type'].dropna().unique().tolist() if not property_db.empty else []
    
    prompt = f"""
Extract search parameters from this real estate question: "{user_question}"

Available locations in our database: {locations[:10]}  # Show first 10 to avoid token limit
Available property types: {property_types}

Return a JSON object with these keys (use null if not mentioned):
- "location": string (exact match from available locations, or closest match)
- "property_type": string (from available types)
- "max_budget": number (if budget mentioned)
- "has_parking": boolean (if parking mentioned)

Example: {{"location": "Dubai Marina", "property_type": "Apartment", "max_budget": 5000000, "has_parking": true}}

Return only the JSON object, nothing else.
"""
    
    try:
        result = call_openai_api([{"role": "user", "content": prompt}], temperature=0, max_tokens=200)
        if result:
            return json.loads(result)
        return {}
    except Exception as e:
        print(f"Parameter extraction error: {e}")
        return {}

# --- Answer User Questions Using Database Only ---
def answer_user_question(user_question):
    """Answer questions using only the property database"""
    # First check if the question is real estate related
    if not is_real_estate_related(user_question):
        return "I'm sorry, I don't have that information right now. I'll connect you with a Property Agent at the end of our call who can assist further."
    
    if property_db.empty:
        return "Sorry, I don't have access to the property database right now."
    
    # Extract search parameters
    search_params = extract_search_params(user_question)
    
    # Search for matching properties
    matching_properties = search_properties(search_params)
    
    if not matching_properties:
        return "I couldn't find any properties matching your criteria in our database. Could you try with different requirements?"
    
    # Format the response using AI with database results only
    properties_text = ""
    for i, prop in enumerate(matching_properties, 1):
        properties_text += f"Property {i}:\n"
        # Show key information from your CSV columns
        if 'Location' in prop and pd.notna(prop['Location']):
            properties_text += f"Location: {prop['Location']}\n"
        if 'Building Name' in prop and pd.notna(prop['Building Name']):
            properties_text += f"Building: {prop['Building Name']}\n"
        if 'Property Type' in prop and pd.notna(prop['Property Type']):
            properties_text += f"Type: {prop['Property Type']}\n"
        if 'No. of Rooms' in prop and pd.notna(prop['No. of Rooms']):
            properties_text += f"Rooms: {prop['No. of Rooms']}\n"
        if 'Area (sqft)' in prop and pd.notna(prop['Area (sqft)']):
            properties_text += f"Area: {prop['Area (sqft)']} sqft\n"
        if 'Purchase Price (USD)' in prop and pd.notna(prop['Purchase Price (USD)']):
            properties_text += f"Purchase Price: ${prop['Purchase Price (USD)']:,.2f}\n"
        if 'Monthly Rent (USD)' in prop and pd.notna(prop['Monthly Rent (USD)']):
            properties_text += f"Monthly Rent: ${prop['Monthly Rent (USD)']:,.2f}\n"
        if 'Parking Spots' in prop and pd.notna(prop['Parking Spots']):
            properties_text += f"Parking: {prop['Parking Spots']} spots\n"
        properties_text += "\n"
    
    prompt = f"""
You are Agent Shreyash, a real estate assistant. A user asked: "{user_question}"

Here are the ONLY properties from our database that match their criteria:

{properties_text}

Based ONLY on these database results, provide a helpful response. If no properties match perfectly, mention what's available that's closest to their needs.

IMPORTANT: Only mention properties and details that are explicitly shown in the database results above. Do not invent or assume any property names, locations, or details.

Keep the response conversational and under 150 words.
"""
    
    try:
        result = call_openai_api([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=200)
        return result if result else "Sorry, I ran into an issue processing your question."
    except Exception as e:
        print(f"Answering error: {e}")
        return "Sorry, I ran into an issue processing your question."

# --- Validate Answer Using AI ---
def validate_answer(question, answer):
    """Simple validation without AI to avoid API issues during form filling"""
    # Basic validation rules
    if not answer or len(answer.strip()) < 1:
        return False
    
    # Question-specific validation - made more flexible
    if "name" in question.lower():
        # Name should contain letters and be reasonable length
        return len(answer.strip()) >= 2 and any(c.isalpha() for c in answer)
    
    elif "contact" in question.lower() or "number" in question.lower():
        # Contact should contain digits - more flexible
        return any(c.isdigit() for c in answer) and len(answer.strip()) >= 3
    
    elif "rent or buy" in question.lower():
        # Should contain rent, buy, or purchase - more flexible
        answer_lower = answer.lower()
        return any(word in answer_lower for word in ["rent", "buy", "purchase", "sale", "by", "bye"])
    
    elif "parking" in question.lower():
        # Should contain yes, no, or related words - more flexible
        answer_lower = answer.lower()
        return any(word in answer_lower for word in ["yes", "no", "need", "want", "required", "necessary", "sure", "okay"])
    
    elif "location" in question.lower():
        # Should be a reasonable location name - more flexible
        return len(answer.strip()) >= 2 and any(c.isalpha() for c in answer)
    
    elif "budget" in question.lower():
        # Should contain numbers or budget-related words - more flexible
        return any(c.isdigit() for c in answer) or any(word in answer.lower() for word in ["lakh", "crore", "thousand", "million", "affordable", "expensive", "dirham", "dollar"])
    
    # Default: accept if it's not empty and has reasonable content
    return len(answer.strip()) >= 1

# --- Main Program ---
def main():
    # Check if databases loaded successfully
    if property_db.empty:
        speak("Sorry, I couldn't load the property database. Please make sure propdata.csv is in the same folder.")
        return
    
    if broker_db.empty:
        speak("Warning: I couldn't load the broker database. Please make sure brokdata.csv is in the same folder.")
    
    # --- Start Conversation ---
    speak("Hi! I'm Agent Shreyash, your real estate assistant.")
    speak("I have access to our property database with real listings.")
    speak("How are you today?")
    listen()
    
    # --- Ask if they have questions ---
    speak("Do you have any questions about our available properties?")
    try_count = 0
    while try_count < 3:
        initial_reply = listen().lower()
        if any(word in initial_reply for word in ["yes", "yeah", "sure", "i do", "question"]):
            while True:
                speak("Please ask your question about properties.")
                user_question = listen()
                
                if user_question.strip():
                    answer = answer_user_question(user_question)
                    speak(answer)
                
                speak("Would you like to ask another question?")
                more = listen().lower()
                if any(x in more for x in ["no", "nothing", "that's it", "nope"]):
                    break
            break
        elif any(word in initial_reply for word in ["no", "nope", "not really", "later"]):
            break
        else:
            try_count += 1
            speak("Sorry, I didn't catch that. Could you please say yes or no?")
    
    if try_count >= 3:
        speak("No problem. Let's continue.")
    
    # --- Ask permission to begin the form ---
    speak("Would you like to proceed with filling out a property inquiry form?")
    permission = listen().lower()
    if "no" in permission:
        speak("Alright, you can come back anytime. Have a great day!")
        return
    else:
        speak("Great! Let's begin the form.")
    
    # --- Ask each question in the form ---
    retry_limit = 2  # Maximum retries per question
    for field, question in questions:
        retry_count = 0
        while retry_count <= retry_limit:
            speak(question)
            answer = listen()
            
            if answer and validate_answer(question, answer):
                valid_answers[field] = answer
                # Confirmation for critical fields
                if field in ["Name", "Contact"]:
                    speak(f"Got it, {answer}. Let's continue.")
                break
            else:
                retry_count += 1
                if retry_count <= retry_limit:
                    speak("That didn't seem clear. Could you please say that again?")
                else:
                    # After max retries, ask them to spell it out or provide a fallback
                    speak("I'm having trouble understanding. Let me skip this for now and we'll have our agent confirm this with you later.")
                    valid_answers[field] = answer if answer else "To be confirmed by agent"
                    break
    
    # --- Find appropriate broker based on user's location preference ---
    user_location = valid_answers.get('Location', '')
    assigned_broker = find_matching_broker(user_location)
    
    if assigned_broker is not None:
        # Add broker information to user data
        valid_answers['Assigned_Broker_Name'] = assigned_broker.get('Agent Name', 'N/A')
        valid_answers['Assigned_Broker_Phone'] = assigned_broker.get('Phone Number', 'N/A')
        valid_answers['Assigned_Broker_Email'] = assigned_broker.get('Email', 'N/A')
        valid_answers['Assigned_Broker_Location'] = assigned_broker.get('Location', 'N/A')
        
        # Announce broker assignment
        broker_info = f"Based on your location preference, I'm assigning you to {assigned_broker.get('Agent Name', 'our agent')}."
        if pd.notna(assigned_broker.get('Phone Number')):
            broker_info += f" You can reach them at {assigned_broker['Phone Number']}."
        if pd.notna(assigned_broker.get('Email')):
            broker_info += f" Their email is {assigned_broker['Email']}."
        
        speak(broker_info)
    else:
        # No broker found
        valid_answers['Assigned_Broker_Name'] = 'No broker assigned'
        valid_answers['Assigned_Broker_Phone'] = 'N/A'
        valid_answers['Assigned_Broker_Email'] = 'N/A'
        valid_answers['Assigned_Broker_Location'] = 'N/A'
        speak("I'll have one of our property agents contact you soon.")
    
    # --- Save the responses to Excel ---
    df = pd.DataFrame([valid_answers])
    excel_file = "real_estate_voice_leads.xlsx"
    
    try:
        if os.path.exists(excel_file):
            # Use openpyxl engine to avoid xlrd compatibility issues
            existing = pd.read_excel(excel_file, engine='openpyxl')
            df = pd.concat([existing, df], ignore_index=True)
        
        # Save with openpyxl engine
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print("✅ Data saved successfully to Excel file")
        
    except Exception as e:
        print(f"❌ Error saving to Excel: {e}")
        # Fallback to CSV if Excel fails
        csv_file = "real_estate_voice_leads.csv"
        try:
            if os.path.exists(csv_file):
                existing = pd.read_csv(csv_file)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_csv(csv_file, index=False)
            print("✅ Data saved to CSV file as fallback")
        except Exception as csv_error:
            print(f"❌ Error saving to CSV: {csv_error}")
    
    speak("Thank you! I've saved your details and assigned you to the appropriate property agent. They will contact you soon with matching properties from our database. Have a great day!")

if __name__ == "__main__":
    main()