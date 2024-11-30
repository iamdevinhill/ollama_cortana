import speech_recognition as sr
import requests
import logging
import sys
import os
import re
from functools import lru_cache
from dotenv import load_dotenv
from elevenlabs import play
from elevenlabs.client import ElevenLabs
import concurrent.futures
import threading
import queue

# Load environment variables (only once)
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_assistant.log', delay=True),
        logging.StreamHandler(sys.stdout)
    ]
)

class VoiceAssistant:
    def __init__(self, config=None):
        # Use dict.get() for more efficient config handling
        self.config = {
            'wake_word': os.getenv('WAKE_WORD', 'hey cortana'),
            'ollama_url': os.getenv('OLLAMA_URL', 'http://localhost:11434/api/chat'),
            'ollama_model': os.getenv('OLLAMA_MODEL', 'mistral'),
            'max_history_length': int(os.getenv('MAX_HISTORY_LENGTH', 5)),
            'elevenlabs_api_key': os.getenv('ELEVENLABS_API_KEY'),
            'voice_name': os.getenv('VOICE_NAME', 'Rachel')
        }
        
        # Efficiently update config
        if config:
            self.config.update(config)

        # Use a class-level logger for slight performance improvement
        self.logger = logging.getLogger(self.__class__.__name__)

        # Shared resources with thread-safe queue for audio processing
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()

        # Initialize Ollama conversation with thread-safe list
        self.conversation_history = []
        self.conversation_lock = threading.Lock()

        # Initialize ElevenLabs client
        self.elevenlabs_client = None
        self._init_elevenlabs_voice()

    def _init_elevenlabs_voice(self):
        """Optimize ElevenLabs voice client initialization."""
        if not self.config['elevenlabs_api_key']:
            return
        
        try:
            self.elevenlabs_client = ElevenLabs(
                api_key=self.config['elevenlabs_api_key']
            )
        except Exception as e:
            self.logger.warning(f"ElevenLabs client initialization failed: {e}")

    def create_recognizer(self):
        """Create a new recognizer with optimized settings."""
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.5
        return recognizer

    def speak(self, text):
        """Optimize speech synthesis with concurrent execution."""
        if not self.elevenlabs_client:
            self.logger.warning("ElevenLabs client not initialized.")
            return

        def generate_and_play():
            try:
                audio = self.elevenlabs_client.generate(
                    text=text,
                    voice=self.config['voice_name'],
                    model="eleven_multilingual_v2"
                )
                play(audio)
            except Exception as e:
                self.logger.error(f"Speech synthesis error: {e}")

        # Use thread pool for non-blocking speech
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(generate_and_play)

    @lru_cache(maxsize=64)
    def get_ollama_response(self, query):
        """
        Get response from Ollama API with caching.
        
        :param query: User's input query
        :return: Assistant's response
        """
        # Manage conversation history
        with self.conversation_lock:
            self.conversation_history.append({"role": "user", "content": query})
            
            if len(self.conversation_history) > self.config['max_history_length']:
                self.conversation_history.pop(0)

        # Prepare payload for Ollama request
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
            
            # Update conversation history
            with self.conversation_lock:
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
        Robust wake word detection with pre-compiled regex.
        
        :param text: Input text to check for wake word
        :return: Boolean indicating wake word detection
        """
        wake_word = self.config['wake_word']
        # Pre-compile regex for performance
        pattern = re.compile(rf'\b{re.escape(wake_word)}\b', re.IGNORECASE)
        return pattern.search(text) is not None

    def _recognize_audio(self, audio_source, timeout=None):
        """
        Centralized audio recognition method.
        
        :param audio_source: Speech recognition audio source
        :param timeout: Optional timeout for listening
        :return: Recognized text
        """
        # Create a new recognizer for each recognition attempt
        recognizer = self.create_recognizer()
        
        try:
            # Adjust for ambient noise without entering context
            recognizer.adjust_for_ambient_noise(audio_source, duration=1)
            
            # Listen with optional timeout
            if timeout:
                audio = recognizer.listen(audio_source, timeout=timeout)
            else:
                audio = recognizer.listen(audio_source)
            
            # Recognize speech
            return recognizer.recognize_google(audio).lower()
        except sr.WaitTimeoutError:
            self.logger.info("Audio listening timed out")
            raise
        except sr.UnknownValueError:
            self.logger.info("Could not understand audio")
            raise

    def listen_for_input(self):
        """Implement efficient continuous listening with threading."""
        def audio_listener():
            while not self.stop_event.is_set():
                try:
                    # Create a new microphone source each time
                    with sr.Microphone() as source:
                        # Directly use the audio recognition method
                        text = self._recognize_audio(source)
                        self.audio_queue.put(text)
                
                except (sr.UnknownValueError, sr.WaitTimeoutError):
                    continue
                except Exception as e:
                    self.logger.error(f"Listening error: {e}")

        def process_audio():
            while not self.stop_event.is_set():
                try:
                    text = self.audio_queue.get(timeout=1)
                    self.logger.info(f"Heard: {text}")
                    
                    if self._is_wake_word_detected(text):
                        self.speak("Yes, just a moment.")
                        self.process_user_input()

                    if "goodbye cortana" in text:
                        self.speak("Goodbye! Shutting down.")
                        self.stop_event.set()
                        sys.exit()
                
                except queue.Empty:
                    continue

        # Use threading for non-blocking audio processing
        listener_thread = threading.Thread(target=audio_listener)
        processor_thread = threading.Thread(target=process_audio)
        
        listener_thread.start()
        processor_thread.start()

        try:
            listener_thread.join()
            processor_thread.join()
        except KeyboardInterrupt:
            self.stop_event.set()

    def process_user_input(self):
        """Optimize user input processing with improved error handling."""
        try:
            # Create a new microphone source
            with sr.Microphone() as source:
                self.speak("How can I help you?")
                # Use the centralized recognition method
                user_query = self._recognize_audio(source, timeout=5)
            
            self.logger.info(f"User query: {user_query}")
                
            # Use concurrent execution for response generation and speaking
            with concurrent.futures.ThreadPoolExecutor() as executor:
                response_future = executor.submit(self.get_ollama_response, user_query)
                response = response_future.result()
                executor.submit(self.speak, response)
        
        except sr.WaitTimeoutError:
            self.speak("Listening timeout. Please try again.")
        except sr.UnknownValueError:
            self.speak("Sorry, I couldn't understand what you said.")
        except Exception as e:
            self.logger.error(f"Input processing error: {e}")
            self.speak("An unexpected error occurred. Please try again.")

    def run(self):
        """Comprehensive run method with enhanced error handling."""
        try:
            self.logger.info("Voice Assistant Initialized")
            self.speak("Hi, I am your Cortana AI assistant.")
            self.listen_for_input()
        except KeyboardInterrupt:
            self.speak("Assistant manually stopped.")
        except Exception as e:
            self.logger.critical(f"Critical error in voice assistant: {e}")
            self.speak("A critical error occurred. The assistant will now shut down.")
        finally:
            # Ensure clean shutdown
            self.stop_event.set()

def main():
    try:
        assistant = VoiceAssistant()
        assistant.run()
    except Exception as e:
        print(f"Failed to start voice assistant: {e}")

if __name__ == "__main__":
    main()