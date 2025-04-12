import os
from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__) # Initialize Flask app

class GroqChatbot:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.history = []
        self.system_msg = {"role": "system", "content": "You're a helpful assistant"}

    def get_response(self, prompt: str):
        """Get response from Groq API"""
        self.history.append({"role": "user", "content": prompt})
        
        try:
            # Create a chat completion (non-streaming for web interface)
            response = self.client.chat.completions.create(
                messages=[self.system_msg] + self.history[-10:],
                model="llama3-70b-8192",
                temperature=0.5,
                max_tokens=512,
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Add to history
            self.history.append({"role": "assistant", "content": response_content})
            
            return response_content
            
        except Exception as e:
            self.history.pop()  # Remove failed user message from history
            return f"Error: {str(e)}"

# Create a global instance of the chatbot
chatbot = GroqChatbot()

# Flask routes 
@app.route('/')
def index():
    """Render the chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat API requests"""
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'response': 'Empty message received'})
    
    # Get response from chatbot
    bot_response = chatbot.get_response(user_message)
    
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)