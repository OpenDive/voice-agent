import asyncio
import logging
import os
from typing import Optional
from dotenv import load_dotenv
import pvporcupine
from pvrecorder import PvRecorder
import threading

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, function_tool
from livekit.plugins import openai, deepgram, cartesia, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class WakeWordVoiceAssistant(Agent):
    """
    A LiveKit voice assistant that uses wake word detection to activate conversations.
    Integrates Porcupine wake word detection with LiveKit agents framework.
    """
    
    def __init__(self):
        """Initialize the wake word voice assistant."""
        super().__init__(instructions="You are a helpful voice assistant. You can help with questions, tasks, and conversation. Keep your responses concise and natural for voice interaction.")
        
        # Wake word detection setup
        self.porcupine_access_key = os.getenv("PORCUPINE_ACCESS_KEY")
        self.porcupine = None
        self.recorder = None
        self.wake_word_thread = None
        self.is_monitoring = False
        self.session = None
        
        logger.info("Wake Word Voice Assistant initialized")
    
    async def on_enter(self):
        """Called when the agent enters a session."""
        logger.info("Voice assistant entered session")
        
        # Start wake word detection
        if self.porcupine_access_key:
            await self.start_wake_word_detection()
        else:
            logger.warning("No Porcupine access key provided, skipping wake word detection")
            # Generate initial greeting
            await self.session.generate_reply(
                instructions="Greet the user and let them know you're ready to help."
            )
    
    async def on_exit(self):
        """Called when the agent exits a session."""
        logger.info("Voice assistant exiting session")
        await self.stop_wake_word_detection()
    
    async def start_wake_word_detection(self):
        """Start wake word detection in a separate thread."""
        try:
            self.porcupine = pvporcupine.create(
                access_key=self.porcupine_access_key,
                keywords=["hey computer", "hey assistant"],
                # You can also use custom wake word files:
                # keyword_paths=["./wake_words/Hey-Coffee-Bot_en_linux_v3_0_0.ppn"]
            )
            
            self.recorder = PvRecorder(
                device_index=-1,  # Use default microphone
                frame_length=self.porcupine.frame_length
            )
            
            self.is_monitoring = True
            self.wake_word_thread = threading.Thread(target=self._monitor_wake_words, daemon=True)
            self.wake_word_thread.start()
            
            logger.info("Wake word detection started. Say 'hey computer' or 'hey assistant' to activate.")
            
        except Exception as e:
            logger.error(f"Failed to start wake word detection: {e}")
    
    def _monitor_wake_words(self):
        """Monitor for wake words in a separate thread."""
        try:
            self.recorder.start()
            
            while self.is_monitoring:
                try:
                    pcm = self.recorder.read()
                    keyword_index = self.porcupine.process(pcm)
                    
                    if keyword_index >= 0:
                        logger.info("Wake word detected! Activating voice assistant...")
                        # Trigger conversation using asyncio
                        if self.session:
                            asyncio.run_coroutine_threadsafe(
                                self._handle_wake_word(), 
                                asyncio.get_event_loop()
                            )
                        
                except Exception as e:
                    logger.error(f"Error processing wake word audio: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Wake word monitoring error: {e}")
        finally:
            if self.recorder:
                self.recorder.stop()
    
    async def _handle_wake_word(self):
        """Handle wake word detection and start conversation."""
        try:
            # Generate a response to acknowledge the wake word
            await self.session.generate_reply(
                instructions="The user has said your wake word. Acknowledge that you heard them and ask how you can help."
            )
        except Exception as e:
            logger.error(f"Error handling wake word: {e}")
    
    async def stop_wake_word_detection(self):
        """Stop wake word detection and cleanup resources."""
        self.is_monitoring = False
        
        if self.wake_word_thread:
            self.wake_word_thread.join(timeout=2)
        
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
            
        if self.recorder:
            self.recorder.delete()
            self.recorder = None
        
        logger.info("Wake word detection stopped")

    @function_tool
    async def get_current_time(self) -> str:
        """Get the current time."""
        import datetime
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}"

    @function_tool
    async def get_current_date(self) -> str:
        """Get the current date."""
        import datetime
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        return f"Today's date is {current_date}"


class StandardVoiceAssistant(Agent):
    """
    A standard LiveKit voice assistant without wake word detection.
    Use this for always-on voice interaction.
    """
    
    def __init__(self):
        super().__init__(instructions="You are a helpful voice assistant. You can help with questions, tasks, and conversation. Keep your responses concise and natural for voice interaction.")
        logger.info("Standard Voice Assistant initialized")
    
    async def on_enter(self):
        """Called when the agent enters a session."""
        logger.info("Voice assistant entered session")
        await self.session.generate_reply(
            instructions="Greet the user warmly and let them know you're ready to help with any questions or tasks."
        )

    @function_tool
    async def get_current_time(self) -> str:
        """Get the current time."""
        import datetime
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}"

    @function_tool
    async def get_current_date(self) -> str:
        """Get the current date."""
        import datetime
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        return f"Today's date is {current_date}"


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the LiveKit voice agent.
    This function is called when a new job is dispatched to the worker.
    """
    await ctx.connect()
    
    # Check if wake word detection is enabled
    use_wake_word = os.getenv("PORCUPINE_ACCESS_KEY") is not None
    
    if use_wake_word:
        agent = WakeWordVoiceAssistant()
        logger.info("Using wake word detection mode")
    else:
        agent = StandardVoiceAssistant()
        logger.info("Using standard always-on mode")
    
    # Create the agent session with the AI pipeline
    # You can choose between two approaches:
    
    # Option 1: Use OpenAI Realtime API (recommended for low latency)
    if os.getenv("USE_REALTIME_API", "true").lower() == "true":
        session = AgentSession(
            llm=openai.realtime.RealtimeModel(
                model="gpt-4o-realtime-preview",
                voice="alloy",
                temperature=0.8,
            ),
            vad=silero.VAD.load(),
        )
        logger.info("Using OpenAI Realtime API")
    
    # Option 2: Use traditional STT-LLM-TTS pipeline
    else:
        session = AgentSession(
            stt=deepgram.STT(model="nova-3", language="multi"),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )
        logger.info("Using STT-LLM-TTS pipeline")
    
    # Store session reference for wake word agent
    if hasattr(agent, 'session'):
        agent.session = session
    
    # Start the session with noise cancellation if available
    room_input_options = None
    if os.getenv("LIVEKIT_API_KEY"):  # Only use if LiveKit Cloud is configured
        try:
            from livekit.agents import RoomInputOptions
            room_input_options = RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            )
            logger.info("Noise cancellation enabled")
        except ImportError:
            logger.info("Noise cancellation not available")
    
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=room_input_options,
    )
    
    logger.info("Voice agent session started successfully")


def main():
    """Main function to run the voice agent."""
    
    # Validate required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please check your .env file and ensure all required API keys are set.")
        return
    
    # Check optional variables
    if not os.getenv("PORCUPINE_ACCESS_KEY"):
        logger.warning("PORCUPINE_ACCESS_KEY not set. Wake word detection will be disabled.")
    
    if not os.getenv("DEEPGRAM_API_KEY") and os.getenv("USE_REALTIME_API", "true").lower() != "true":
        logger.warning("DEEPGRAM_API_KEY not set. STT-LLM-TTS pipeline will not work.")
    
    if not os.getenv("CARTESIA_API_KEY") and os.getenv("USE_REALTIME_API", "true").lower() != "true":
        logger.warning("CARTESIA_API_KEY not set. STT-LLM-TTS pipeline will not work.")
    
    logger.info("Starting LiveKit Voice Agent...")
    logger.info("Available modes:")
    logger.info("  python livekit.py console  - Run in terminal mode")
    logger.info("  python livekit.py dev      - Run in development mode")
    logger.info("  python livekit.py start    - Run in production mode")
    
    # Run the agent with LiveKit CLI
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    main()
