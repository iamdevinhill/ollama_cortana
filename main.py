import speech_recognition as sr
import pyttsx3
import requests
import logging
import sys

class VoiceAssistant:
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize the recognizer and microphone
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize the speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

        # Get available voices and set the desired one
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)  # Use voices[1] for the second voice in the list

        # Ollama API settings
        self.ollama_url = 'http://localhost:11434/api/chat'
        self.model = "mistral"

        # Conversation history settings
        self.conversation_history = []
        self.max_history_length = 5


    def speak(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Speech synthesis error: {e}")

    def get_ollama_response(self, query):
        self.conversation_history.append({"role": "user", "content": query})
        
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history.pop(0)

        payload = {
            "model": self.model,
            "messages": self.conversation_history,
            "stream": False
        }
        
        try:
            response = requests.post(self.ollama_url, 
                                     json=payload, 
                                     timeout=10)
            response.raise_for_status()
            assistant_response = response.json()['message']['content']
            
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
        
        except requests.RequestException as e:
            self.logger.error(f"Ollama request error: {e}")
            return "Sorry, I'm experiencing technical difficulties."

    def listen_for_input(self, wake_word="hey cortana"):
        while True:
            try:
                audio = None
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    self.logger.info("Listening for wake word...")
                    audio = self.recognizer.listen(source)
                
                if audio:
                    text = self.recognizer.recognize_google(audio).lower()
                    self.logger.info(f"Heard: {text}")
                    
                    if wake_word in text:
                        self.speak("Hey there.")
                        self.process_user_input()

                    if "goodbye cortana" in text:
                        self.speak("Goodbye! Shutting down.")
                        sys.exit()  # Exit the script
            
            except sr.UnknownValueError:
                continue
            except Exception as e:
                self.logger.error(f"Listening error: {e}")

    def process_user_input(self):
        try:
            audio = None
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.speak("How can I help?")
                audio = self.recognizer.listen(source, timeout=5)
            
            if audio:
                user_query = self.recognizer.recognize_google(audio)
                self.logger.info(f"User query: {user_query}")
                
                response = self.get_ollama_response(user_query)
                self.speak(response)
        
        except sr.WaitTimeoutError:
            self.speak("Listening timeout. Please try again.")
        except sr.UnknownValueError:
            self.speak("Sorry, I couldn't understand that.")
        except Exception as e:
            self.logger.error(f"Input processing error: {e}")
            self.speak("An unexpected error occurred.")

    def run(self):
        try:
            self.logger.info("Voice Assistant Initialized")
            self.speak("Hi, your voice assistant is ready.")
            self.listen_for_input()
        except Exception as e:
            self.logger.critical(f"Critical error in voice assistant: {e}")

def main():
    assistant = VoiceAssistant()
    assistant.run()

if __name__ == "__main__":
    main()