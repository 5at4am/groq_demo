import os # Importing os for environment variable management
from groq import Groq # Importing Groq for API interaction
from dotenv import load_dotenv  # Importing load_dotenv for loading environment variables

load_dotenv()   # Load environment variables from .env file

class GroqChatbot:  # Class to encapsulate the chatbot functionality
    def __init__(self): # Constructor to initialize the chatbot
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))   # Initialize Groq client with API key from environment variable
        self.history = []   # List to store conversation history
        self.system_msg = {"role": "system", "content": "You're a helpful assistant"} # System message to set the context for the assistant

    def _stream_response(self, prompt: str): # Private method to handle streaming response
        """Handle streaming response with proper error handling"""  # Docstring for the method
        self.history.append({"role": "user", "content": prompt})    # Append user message to history
        
        try:    # Attempt to get a response from the API
            stream = self.client.chat.completions.create( # Create a chat completion stream
                messages=[self.system_msg] + self.history[-10:],    # Include system message and last 10 messages from history
                model="llama3-70b-8192",    # Specify the model to use
                temperature=0.5,    # Set temperature for randomness in responses
                max_tokens=512, # Set maximum tokens for the response
                stream=True,    # Enable streaming for real-time response
            )   

            full_response = []  # List to store full response
            print("\nAssistant: ", end="", flush=True)  # Print assistant prefix
            for chunk in stream:    # Iterate over the streamed chunks
                if content := chunk.choices[0].delta.content:   # Extract content from the chunk
                    print(content, end="", flush=True)  # Print the content in real-time
                    full_response.append(content)   # Append content to full response list
            
            self.history.append({"role": "assistant", "content": "".join(full_response)})    # Append full response to history
            
        except Exception as e:  # Catch any exceptions during the API call
            print(f"\n⚠️ Error: {str(e)}")  # Print error message
            self.history.pop()  # Remove failed user message from history

    def run(self):  # Main method to run the chatbot
        """Improved main loop with cleaner exit handling""" # Docstring for the method
        print("Chatbot ready. Type 'exit' to quit\n")   # Print initial message
        while True: # Infinite loop for continuous interaction
            try:    # Attempt to get user input
                # Prompt user for input and check for exit commands
                if (user_input := input("\nYou: ").lower()) in {'exit', 'quit'}:    # Check for exit commands
                    print("\nGoodbye! 👋")  # print 
                    break   # Exit the loop if exit command is given
                self._stream_response(user_input)   # Call the method to get a response from the assistant
                
            except KeyboardInterrupt:   # Catch keyboard interrupt (Ctrl+C)
                print("\nSession ended by user")    # Print message for session end
                break   # Exit the loop on keyboard interrupt

if __name__ == "__main__":  # Main entry point of the script
    GroqChatbot().run()  # Single instance call to run the chatbot
    # This will start the chatbot and allow user interaction
    # The chatbot will keep running until the user types 'exit' or 'quit' 
    # or presses Ctrl+C to interrupt the session.
    # The chatbot will also print a message when the session is ended.
    # The chatbot will also print a message when an error occurs during the API call.
    # The chatbot will also print a message when the session is ended by the user.