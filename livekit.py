import asyncio
import logging
import os
import json
from typing import Optional
from dotenv import load_dotenv
import pvporcupine
from pvrecorder import PvRecorder
import numpy as np
import sounddevice as sd
from openai import AsyncOpenAI
from livekit import api, rtc, agents
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai as lk_openai, silero
import threading
import queue
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class LiveKitVoiceAgent:
    def __init__(self):
        """Initialize the LiveKit Voice Agent with OpenAI integration."""
        
        # API Keys and configuration
        self.porcupine_access_key = os.getenv("PORCUPINE_ACCESS_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.livekit_api_key = os.getenv("LIVEKIT_API_KEY")
        self.livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
        self.livekit_url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
        
        # Validate required environment variables
        if not all([self.porcupine_access_key, self.openai_api_key]):
            raise ValueError("Missing required environment variables. Please check your .env file.")
        
        # Initialize components
        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        self.porcupine = None
        self.recorder = None
        self.room = None
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.conversation_active = False
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        
        logger.info("LiveKit Voice Agent initialized")

    async def setup_wake_word_detection(self):
        """Setup Porcupine wake word detection."""
        try:
            self.porcupine = pvporcupine.create(
                access_key=self.porcupine_access_key,
                keywords=["hey computer", "hey assistant"],  # Built-in keywords
                # You can also use custom wake word files:
                # keyword_paths=["./wake_words/Hey-Coffee-Bot_en_linux_v3_0_0.ppn"]
            )
            
            self.recorder = PvRecorder(
                device_index=-1,  # Use default microphone
                frame_length=self.porcupine.frame_length
            )
            
            logger.info("Wake word detection setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup wake word detection: {e}")
            return False

    async def setup_livekit_room(self, room_name: str = "voice_agent_room") -> bool:
        """Setup and connect to LiveKit room."""
        try:
            if not self.livekit_api_key or not self.livekit_api_secret:
                logger.warning("LiveKit credentials not provided, running in local mode")
                return True
                
            # Create room token
            token = api.AccessToken(self.livekit_api_key, self.livekit_api_secret) \
                .with_identity("voice_agent") \
                .with_name("Voice Agent") \
                .with_grants(api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True
                )).to_jwt()
            
            # Connect to room
            self.room = rtc.Room()
            
            # Setup event handlers
            @self.room.on("participant_connected")
            def on_participant_connected(participant: rtc.RemoteParticipant):
                logger.info(f"Participant connected: {participant.identity}")
            
            @self.room.on("track_subscribed")
            def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"Subscribed to audio track from {participant.identity}")
                    
            await self.room.connect(self.livekit_url, token)
            logger.info(f"Connected to LiveKit room: {room_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup LiveKit room: {e}")
            return False

    def start_wake_word_monitoring(self):
        """Start monitoring for wake words in a separate thread."""
        def monitor_wake_words():
            try:
                self.recorder.start()
                logger.info("Wake word monitoring started. Say 'hey computer' or 'hey assistant'")
                
                while self.is_listening:
                    try:
                        pcm = self.recorder.read()
                        keyword_index = self.porcupine.process(pcm)
                        
                        if keyword_index >= 0:
                            logger.info("Wake word detected! Starting conversation...")
                            asyncio.run_coroutine_threadsafe(
                                self.handle_wake_word_detected(), 
                                asyncio.get_event_loop()
                            )
                            
                    except Exception as e:
                        logger.error(f"Error processing audio: {e}")
                        break
                        
            except Exception as e:
                logger.error(f"Wake word monitoring error: {e}")
            finally:
                if self.recorder:
                    self.recorder.stop()
                    
        # Start monitoring in a separate thread
        self.wake_word_thread = threading.Thread(target=monitor_wake_words, daemon=True)
        self.wake_word_thread.start()

    async def handle_wake_word_detected(self):
        """Handle wake word detection and start conversation."""
        if self.conversation_active:
            logger.info("Conversation already active, ignoring wake word")
            return
            
        self.conversation_active = True
        logger.info("Starting voice conversation...")
        
        try:
            # Start audio recording for conversation
            await self.start_conversation()
            
        except Exception as e:
            logger.error(f"Error in conversation: {e}")
        finally:
            self.conversation_active = False
            logger.info("Conversation ended")

    async def start_conversation(self):
        """Start a conversation session with OpenAI."""
        try:
            # Record audio for conversation
            logger.info("Listening for your question... (speak now)")
            audio_data = await self.record_conversation_audio(duration=5)  # 5 seconds
            
            if audio_data is not None:
                # Convert audio to text using OpenAI Whisper
                text = await self.speech_to_text(audio_data)
                
                if text:
                    logger.info(f"You said: {text}")
                    
                    # Get response from OpenAI
                    response = await self.get_openai_response(text)
                    
                    if response:
                        logger.info(f"Assistant: {response}")
                        
                        # Convert text to speech and play
                        await self.text_to_speech_and_play(response)
                    
        except Exception as e:
            logger.error(f"Conversation error: {e}")

    async def record_conversation_audio(self, duration: int = 5) -> Optional[np.ndarray]:
        """Record audio for conversation."""
        try:
            logger.info(f"Recording for {duration} seconds...")
            
            # Record audio using sounddevice
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32
            )
            sd.wait()  # Wait for recording to complete
            
            # Convert to 16-bit PCM
            audio_data = (audio_data * 32767).astype(np.int16)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
            return None

    async def speech_to_text(self, audio_data: np.ndarray) -> Optional[str]:
        """Convert speech to text using OpenAI Whisper."""
        try:
            # Convert numpy array to bytes
            audio_bytes = audio_data.tobytes()
            
            # Create a temporary audio file-like object
            import io
            import wave
            
            audio_buffer = io.BytesIO()
            with wave.open(audio_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_bytes)
            
            audio_buffer.seek(0)
            audio_buffer.name = "audio.wav"  # Required by OpenAI API
            
            # Transcribe using OpenAI Whisper
            transcript = await self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_buffer,
                response_format="text"
            )
            
            return transcript.strip() if transcript else None
            
        except Exception as e:
            logger.error(f"Speech-to-text error: {e}")
            return None

    async def get_openai_response(self, text: str) -> Optional[str]:
        """Get response from OpenAI ChatGPT."""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful voice assistant. Keep your responses concise and conversational, suitable for spoken interaction."
                    },
                    {"role": "user", "content": text}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None

    async def text_to_speech_and_play(self, text: str):
        """Convert text to speech using OpenAI TTS and play it."""
        try:
            # Generate speech using OpenAI TTS
            response = await self.openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="wav"
            )
            
            # Play the audio
            import io
            import soundfile as sf
            
            audio_buffer = io.BytesIO(response.content)
            audio_data, sample_rate = sf.read(audio_buffer)
            
            # Play audio
            sd.play(audio_data, sample_rate)
            sd.wait()  # Wait for playback to complete
            
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            # Fallback: just print the response
            print(f"Assistant (text): {text}")

    async def start(self):
        """Start the voice agent."""
        logger.info("Starting LiveKit Voice Agent...")
        
        # Setup components
        if not await self.setup_wake_word_detection():
            logger.error("Failed to setup wake word detection")
            return
            
        # Setup LiveKit room (optional)
        await self.setup_livekit_room()
        
        # Start listening for wake words
        self.is_listening = True
        self.start_wake_word_monitoring()
        
        logger.info("Voice agent is running. Say 'hey computer' or 'hey assistant' to start.")
        logger.info("Press Ctrl+C to stop.")
        
        try:
            # Keep the main thread alive
            while self.is_listening:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Stopping voice agent...")
            await self.stop()

    async def stop(self):
        """Stop the voice agent and cleanup resources."""
        logger.info("Shutting down voice agent...")
        
        self.is_listening = False
        
        # Cleanup Porcupine resources
        if self.porcupine:
            self.porcupine.delete()
            
        if self.recorder:
            self.recorder.delete()
            
        # Disconnect from LiveKit room
        if self.room:
            await self.room.disconnect()
            
        logger.info("Voice agent stopped")


# Example usage and main function
async def main():
    """Main function to run the voice agent."""
    agent = LiveKitVoiceAgent()
    
    try:
        await agent.start()
    except Exception as e:
        logger.error(f"Error running voice agent: {e}")
    finally:
        await agent.stop()


if __name__ == "__main__":
    # Run the voice agent
    asyncio.run(main())
