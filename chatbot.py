import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv() 

class GroqChatbot:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # No spaces in key name
        self.history = []
        self.system_msg = {"role": "system", "content": "You're a helpful assistant"}

    def _get_api_key(self):
        """Load API key from environment"""
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("Missing GROQ_API_KEY in .env file")
        return key

    def _stream_response(self, prompt: str):
        """Stream API response"""
        self.history.append({"role": "user", "content": prompt})
        
        stream = self.client.chat.completions.create(
            messages=[self.system_msg] + self.history[-10:],  # Keep last 10 messages
          model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=512,
            stream=True,
        )

        full_response = []
        print("\nAssistant: ", end="")
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                full_response.append(content)
        
        self.history.append({"role": "assistant", "content": "".join(full_response)})

    def run(self):
        """Main chat interface"""
        print("Chatbot ready. Type 'exit' to quit\n")
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                self._stream_response(user_input)
            except KeyboardInterrupt:
                print("\nSession ended")
                break

if __name__ == "__main__":
    chatbot = GroqChatbot()
    chatbot.run()
    GroqChatbot().run()