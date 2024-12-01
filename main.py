import speech_recognition as sr
import requests
import logging
import sys
import os
import re
from dotenv import load_dotenv
from elevenlabs import play
from elevenlabs.client import ElevenLabs

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_assistant.log'),
        logging.StreamHandler()
    ]
)

class VoiceAssistant:
    def __init__(self, config=None):
        # Default configuration with option to override
        self.config = {
            'wake_word': os.getenv('WAKE_WORD', 'hey cortana'),
            'ollama_url': os.getenv('OLLAMA_URL', 'http://localhost:11434/api/chat'),
            'ollama_model': os.getenv('OLLAMA_MODEL', 'mistral'),
            'max_history_length': int(os.getenv('MAX_HISTORY_LENGTH', 5)),
            'elevenlabs_api_key': os.getenv('ELEVENLABS_API_KEY'),
            'voice_name': os.getenv('VOICE_NAME', 'Rachel')
        }
        
        # Override with provided config if any
        if config:
            self.config.update(config)

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Initialize the recognizer and microphone
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize ElevenLabs voice client
        self._init_elevenlabs_voice()

        # Initialize Ollama conversation
        self.conversation_history = []

    def _init_elevenlabs_voice(self):
        """Initialize ElevenLabs voice client."""
        if self.config['elevenlabs_api_key']:
            try:
                self.elevenlabs_client = ElevenLabs(
                    api_key=self.config['elevenlabs_api_key']
                )
            except Exception as e:
                self.logger.warning(f"ElevenLabs client initialization failed: {e}")
                self.elevenlabs_client = None

    def speak(self, text):
        """Speak the given text using ElevenLabs."""
        try:
            if self.elevenlabs_client:
                # Generate audio using ElevenLabs
                audio = self.elevenlabs_client.generate(
                    text=text,
                    voice=self.config['voice_name'],
                    model="eleven_multilingual_v2"
                )
                # Play the generated audio
                play(audio)
            else:
                self.logger.warning("ElevenLabs client not initialized. Cannot speak.")
        except Exception as e:
            self.logger.error(f"Speech synthesis error: {e}")

    def get_ollama_response(self, query):
        """Get response from Ollama API."""
        # Manage conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        if len(self.conversation_history) > self.config['max_history_length']:
            self.conversation_history.pop(0)

        payload = {
            "model": self.config['ollama_model'],
            "messages": self.conversation_history,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.config['ollama_url'], 
                json=payload, 
                timeout=10
            )
            response.raise_for_status()
            assistant_response = response.json()['message']['content']
            
            self.conversation_history.append({
                "role": "assistant", 
                "content": assistant_response
            })
            
            return assistant_response
        
        except requests.RequestException as e:
            self.logger.error(f"Ollama request error: {e}")
            return "Sorry, I'm experiencing technical difficulties connecting to my AI system."

    def _is_wake_word_detected(self, text):
        """
        More robust wake word detection with regex.
        Allows for some variation in wake word pronunciation.
        """
        wake_word = self.config['wake_word']
        pattern = rf'\b{re.escape(wake_word)}\b'
        return re.search(pattern, text.lower()) is not None

    def listen_for_input(self):
        """Continuously listen for wake word and process input."""
        while True:
            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    self.logger.info("Listening for wake word...")
                    audio = self.recognizer.listen(source)
                
                if audio:
                    text = self.recognizer.recognize_google(audio).lower()
                    self.logger.info(f"Heard: {text}")
                    
                    if self._is_wake_word_detected(text):
                        self.speak("Yes, just a moment.")
                        self.process_user_input()

                    if "goodbye cortana" in text:
                        self.speak(f"Goodbye! Shutting down.")
                        sys.exit()
            
            except sr.UnknownValueError:
                continue
            except Exception as e:
                self.logger.error(f"Listening error: {e}")

    def process_user_input(self):
        """Process user input with enhanced error handling."""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.speak("How can I help you?")
                audio = self.recognizer.listen(source, timeout=5)
            
            user_query = self.recognizer.recognize_google(audio)
            self.logger.info(f"User query: {user_query}")
            self.speak("Let me check.")
                
            response = self.get_ollama_response(user_query)
            self.logger.info(f"Response: {response}")
            self.speak(response)
        
        except sr.WaitTimeoutError:
            self.speak("Listening timeout. Please try again.")
        except sr.UnknownValueError:
            self.speak("Sorry, I couldn't understand what you said.")
        except Exception as e:
            self.logger.error(f"Input processing error: {e}")
            self.speak("An unexpected error occurred. Please try again.")

    def run(self):
        """Main run method with comprehensive error handling."""
        try:
            self.logger.info("Voice Assistant Initialized")
            self.speak(f"Hi, your Cortana assistant is ready.")
            self.listen_for_input()
        except KeyboardInterrupt:
            self.speak("Assistant manually stopped.")
        except Exception as e:
            self.logger.critical(f"Critical error in voice assistant: {e}")
            self.speak("A critical error occurred. The assistant will now shut down.")

def main():
    try:
        assistant = VoiceAssistant()
        assistant.run()
    except Exception as e:
        print(f"Failed to start voice assistant: {e}")

if __name__ == "__main__":
    main()