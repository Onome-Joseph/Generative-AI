import os
import json
import time
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import base64

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Store user sessions
user_sessions = {}

class SimpleAITutor:
    """Simplified AI tutor without LangChain dependencies"""
    
    def __init__(self, language="English", proficiency="beginner"):
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.language = language
        self.proficiency = proficiency
        
        # Simple conversation memory (store last 5 exchanges)
        self.memory = []
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
    
    def get_system_prompt(self):
        """Generate system prompt based on language and proficiency"""
        prompt = f"""You are Lingo AI, a friendly AI language tutor for {self.language} at {self.proficiency} level.

Your role:
1. Help users learn {self.language}
2. Correct grammar and vocabulary errors gently
3. Provide explanations for language rules
4. Have natural conversations
5. Adapt to the user's proficiency level ({self.proficiency})

Guidelines:
- Keep responses concise (max 3-4 sentences)
- Use simple language for beginners, more complex for advanced
- Always provide translations when introducing new words
- Give examples to illustrate points
- Be encouraging and positive

Current conversation context: {self.get_memory_context()}"""
        return prompt
    
    def get_memory_context(self):
        """Get recent conversation context"""
        if not self.memory:
            return "This is the start of the conversation."
        
        context = ""
        for i, exchange in enumerate(self.memory[-3:]):  # Last 3 exchanges
            context += f"\nUser: {exchange.get('user', '')}"
            context += f"\nAI: {exchange.get('ai', '')}"
        return context
    
    def add_to_memory(self, user_message, ai_response):
        """Store conversation in memory"""
        self.memory.append({
            'user': user_message,
            'ai': ai_response
        })
        # Keep only last 10 exchanges
        if len(self.memory) > 10:
            self.memory = self.memory[-10:]
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory = []
    
    def get_response(self, user_message):
        """Get AI response using Groq API directly"""
        try:
            # Import Groq here to avoid issues
            from groq import Groq
            
            client = Groq(api_key=self.groq_api_key)
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": self.get_system_prompt()
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
            
            # Call Groq API
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.7,
                max_tokens=300,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            # Extract response
            ai_response = completion.choices[0].message.content
            
            # Store in memory
            self.add_to_memory(user_message, ai_response)
            
            return ai_response.strip()
            
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return f"I'm having trouble responding. Please try again. (Error: {str(e)})"

class SimpleTTS:
    """Simplified Text-to-Speech using Deepgram"""
    
    def __init__(self, language="English"):
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        self.language = language
        
        if not self.api_key:
            print("Warning: DEEPGRAM_API_KEY not set. TTS will be disabled.")
    
    def text_to_speech(self, text):
        """Convert text to speech"""
        if not self.api_key or not text.strip():
            return None
        
        # Select model based on language
        models = {
            "English": "aura-asteria-en",
            "Spanish": "aura-2-estrella-es",
            "French": "aura-athena-fr",
            "German": "aura-orion-de"
        }
        
        model = models.get(self.language, "aura-asteria-en")
        
        url = f"https://api.deepgram.com/v1/speak?model={model}"
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Clean text for TTS
        clean_text = self.clean_text(text)
        
        payload = {"text": clean_text}
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode('utf-8')
            else:
                print(f"TTS Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"TTS Error: {e}")
            return None
    
    def clean_text(self, text):
        """Clean text for TTS"""
        import re
        
        # Remove markdown-like formatting
        clean_text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)  # Bold/Italic
        clean_text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', clean_text)  # Underline
        clean_text = re.sub(r'\[.*?\]\(.*?\)', '', clean_text)  # Links
        clean_text = re.sub(r'`.*?`', '', clean_text)  # Code
        
        # Remove excessive whitespace
        clean_text = ' '.join(clean_text.split())
        
        # Limit length
        if len(clean_text) > 500:
            clean_text = clean_text[:497] + "..."
        
        return clean_text.strip()

def get_or_create_session(session_id, language="English", proficiency="beginner"):
    """Get existing session or create new one"""
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            'ai_tutor': SimpleAITutor(language, proficiency),
            'tts': SimpleTTS(language),
            'created_at': time.time()
        }
    return user_sessions[session_id]

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Lingo AI Backend',
        'version': '1.0',
        'python': '3.13'
    })

# Chat endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id')
        language = data.get('language', 'English')
        proficiency = data.get('proficiency', 'beginner')
        use_voice = data.get('use_voice', False)
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Get or create session
        session_data = get_or_create_session(session_id, language, proficiency)
        ai_tutor = session_data['ai_tutor']
        tts = session_data['tts']
        
        # Get AI response
        bot_response = ai_tutor.get_response(user_message)
        
        # Generate audio if requested
        audio_data = None
        if use_voice and tts.api_key:
            audio_data = tts.text_to_speech(bot_response)
        
        return jsonify({
            'session_id': session_id,
            'bot_response': bot_response,
            'audio_data': audio_data,
            'language': language,
            'proficiency': proficiency
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start new session
@app.route('/api/session/new', methods=['POST'])
def new_session():
    """Start a new chat session"""
    try:
        data = request.json
        language = data.get('language', 'English')
        proficiency = data.get('proficiency', 'beginner')
        
        session_id = str(uuid.uuid4())
        session_data = {
            'ai_tutor': SimpleAITutor(language, proficiency),
            'tts': SimpleTTS(language),
            'created_at': time.time()
        }
        user_sessions[session_id] = session_data
        
        # Get welcome message
        welcome_msg = f"Hello! I'm your {language} tutor. I'll help you practice at the {proficiency} level. What would you like to learn today?"
        
        return jsonify({
            'session_id': session_id,
            'welcome_message': welcome_msg,
            'language': language,
            'proficiency': proficiency,
            'message': 'New session started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Reset conversation
@app.route('/api/session/reset', methods=['POST'])
def reset_session():
    """Reset conversation for current session"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id in user_sessions:
            user_sessions[session_id]['ai_tutor'].clear_memory()
            return jsonify({'message': 'Conversation reset successfully'})
        else:
            return jsonify({'error': 'Session not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Get available languages
@app.route('/api/languages', methods=['GET'])
def get_languages():
    """Get available languages"""
    return jsonify({
        'languages': [
            {'code': 'en', 'name': 'English', 'tts_supported': True},
            {'code': 'es', 'name': 'Spanish', 'tts_supported': True},
            {'code': 'fr', 'name': 'French', 'tts_supported': True},
            {'code': 'de', 'name': 'German', 'tts_supported': True},
            {'code': 'it', 'name': 'Italian', 'tts_supported': False},
            {'code': 'jp', 'name': 'Japanese', 'tts_supported': False},
            {'code': 'zh', 'name': 'Chinese', 'tts_supported': False},
            {'code': 'ko', 'name': 'Korean', 'tts_supported': False}
        ]
    })

# Test endpoint
@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test if API is working"""
    return jsonify({
        'status': 'working',
        'message': 'Lingo AI API is running!',
        'timestamp': time.time()
    })

# Clean up old sessions (optional background task)
def cleanup_old_sessions():
    """Remove sessions older than 2 hours"""
    current_time = time.time()
    old_sessions = []
    
    for session_id, session_data in user_sessions.items():
        if current_time - session_data['created_at'] > 7200:  # 2 hours
            old_sessions.append(session_id)
    
    for session_id in old_sessions:
        del user_sessions[session_id]
    
    if old_sessions:
        print(f"Cleaned up {len(old_sessions)} old sessions")

# Optional: Add a scheduled cleanup (if using a worker process)
# import threading
# def schedule_cleanup():
#     threading.Timer(3600, schedule_cleanup).start()  # Run every hour
#     cleanup_old_sessions()

# if __name__ == '__main__':
#     schedule_cleanup()
#     app.run(debug=True, port=5001)

if __name__ == '__main__':
    # Run cleanup on startup
    cleanup_old_sessions()
    app.run(debug=True, port=5001)
