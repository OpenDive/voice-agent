import asyncio
import logging
import os
import threading
from typing import Optional
from dotenv import load_dotenv
import pvporcupine
from pvrecorder import PvRecorder

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Instructions for the AI assistant
INSTRUCTIONS = """You are a helpful coffee barista robot at a blockchain conference. 

Key behaviors:
- Keep responses concise and natural for voice interaction
- Be friendly and conversational
- Ask follow-up questions when appropriate
- Use the available functions when users ask for time or date
- If wake word detection is enabled, acknowledge when you're activated
- Inquire about coffee preferences

You can help with:
- General questions and conversation
- Telling the current time and date
- Providing assistance with various topics
"""

WELCOME_MESSAGE = "Hello! I'm your coffee barista robot. How can I help you today?"

WAKE_WORD_ACTIVATION_MESSAGE = "Hi! I heard you call me. What can I help you with?"


class VoiceAgentContext:
    """Context manager for voice agent state and wake word detection."""
    
    def __init__(self):
        self.porcupine_access_key = os.getenv("PORCUPINE_ACCESS_KEY")
        self.porcupine = None
        self.recorder = None
        self.wake_word_thread = None
        self.is_monitoring = False
        self.session = None
        self.wake_word_detected = False
        self.conversation_active = False
        self.deactivation_timer = None  # Track current timer task
        self.wake_word_paused = False  # Track if wake word detection is paused
        
    def has_wake_word_capability(self) -> bool:
        """Check if wake word detection is available."""
        return self.porcupine_access_key is not None
    
    def should_respond_to_speech(self) -> bool:
        """Determine if the agent should respond to user speech."""
        if not self.has_wake_word_capability():
            # Always respond if no wake word detection
            return True
        
        # Only respond if wake word was detected or conversation is active
        return self.wake_word_detected or self.conversation_active
    
    async def activate_conversation(self):
        """Activate conversation mode."""
        if not (self.wake_word_detected and self.conversation_active):  # Prevent duplicate activations
            self.wake_word_detected = True
            self.conversation_active = True
            self.wake_word_paused = True  # Pause wake word detection during conversation
            logger.info("Conversation activated - wake word detection paused")
    
    def deactivate_conversation(self):
        """Deactivate conversation mode."""
        self.wake_word_detected = False
        self.conversation_active = False
        self.wake_word_paused = False  # Resume wake word detection
        self.cancel_deactivation_timer()  # Cancel any pending timer
        logger.info("Conversation deactivated - wake word detection resumed")
    
    async def start_wake_word_detection(self):
        """Start wake word detection in a separate thread."""
        if not self.has_wake_word_capability():
            logger.info("Wake word detection not available - running in always-on mode")
            return
            
        try:
            self.porcupine = pvporcupine.create(
                access_key=self.porcupine_access_key,
                keywords=["hey computer", "hey assistant", "hey barista"],
            )
            
            self.recorder = PvRecorder(
                device_index=-1,
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
                    
                    # Skip processing if wake word detection is paused
                    if self.wake_word_paused:
                        continue
                    
                    keyword_index = self.porcupine.process(pcm)
                    
                    if keyword_index >= 0:
                        logger.info("Wake word detected!")
                        
                        # Double-check if conversation is already active (race condition protection)
                        if self.conversation_active:
                            logger.debug("Ignoring wake word - conversation already active")
                            continue
                        
                        # Schedule activation on main thread (thread-safe)
                        try:
                            loop = asyncio.get_event_loop()
                            asyncio.run_coroutine_threadsafe(
                                self._handle_wake_word_activation(),
                                loop
                            )
                        except RuntimeError:
                            logger.warning("Could not schedule wake word activation - event loop not available")
                        
                except Exception as e:
                    logger.error(f"Error processing wake word audio: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Wake word monitoring error: {e}")
        finally:
            if self.recorder:
                self.recorder.stop()
    
    async def _handle_wake_word_activation(self):
        """Handle wake word activation on main thread (thread-safe)."""
        try:
            # Activate conversation (prevents duplicates)
            await self.activate_conversation()
            
            # Send activation message if session is available
            if self.session:
                await self._send_wake_word_response()
                
        except Exception as e:
            logger.error(f"Error handling wake word activation: {e}")

    async def _send_wake_word_response(self):
        """Send wake word activation response."""
        try:
            self.session.conversation.item.create(
                llm.ChatMessage(
                    role="assistant",
                    content=WAKE_WORD_ACTIVATION_MESSAGE
                )
            )
            self.session.response.create()
        except Exception as e:
            logger.error(f"Error sending wake word response: {e}")
    
    async def stop_wake_word_detection(self):
        """Stop wake word detection and cleanup resources."""
        self.is_monitoring = False
        
        # Cancel any pending deactivation timer
        self.cancel_deactivation_timer()
        
        if self.wake_word_thread:
            self.wake_word_thread.join(timeout=2)
        
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
            
        if self.recorder:
            self.recorder.delete()
            self.recorder = None
        
        logger.info("Wake word detection stopped")
    
    def cancel_deactivation_timer(self):
        """Cancel any pending deactivation timer."""
        if self.deactivation_timer and not self.deactivation_timer.done():
            self.deactivation_timer.cancel()
            logger.debug("Deactivation timer cancelled")
    
    def start_deactivation_timer(self, delay: int = 10):
        """Start/restart the conversation deactivation timer."""
        if not self.has_wake_word_capability():
            return  # No timer needed for always-on mode
            
        # Cancel any existing timer
        self.cancel_deactivation_timer()
        
        # Start new timer
        self.deactivation_timer = asyncio.create_task(self._deactivate_after_delay(delay))
        logger.debug(f"Deactivation timer started ({delay}s)")
    
    async def _deactivate_after_delay(self, delay: int):
        """Deactivate conversation after delay."""
        try:
            await asyncio.sleep(delay)
            if self.conversation_active:
                self.deactivate_conversation()
                logger.info("Conversation deactivated due to inactivity")
        except asyncio.CancelledError:
            logger.debug("Deactivation timer was cancelled")
            raise
    
    # Function tools for the assistant
    def get_current_time(self) -> str:
        """Get the current time."""
        current_time = datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}"

    def get_current_date(self) -> str:
        """Get the current date."""
        current_date = datetime.now().strftime("%B %d, %Y")
        return f"Today's date is {current_date}"
    
    def get_current_datetime(self) -> str:
        """Get the current date and time."""
        now = datetime.now()
        date_str = now.strftime("%B %d, %Y")
        time_str = now.strftime("%I:%M %p")
        return f"Today is {date_str} and the current time is {time_str}"


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit voice agent."""
    
    # Connect to the room and wait for participant
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    await ctx.wait_for_participant()
    
    # Initialize voice agent context
    agent_ctx = VoiceAgentContext()
    
    # Create the AI model
    model = openai.realtime.RealtimeModel(
        instructions=INSTRUCTIONS,
        voice=os.getenv("OPENAI_VOICE", "alloy"),  # alloy, echo, fable, onyx, nova, shimmer
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.8")),
        modalities=["audio", "text"]
    )
    
    # Create multimodal assistant
    assistant = MultimodalAgent(model=model, fnc_ctx=agent_ctx)
    assistant.start(ctx.room)
    
    # Get the session for direct management
    session = model.sessions[0]
    agent_ctx.session = session
    
    # Start wake word detection
    await agent_ctx.start_wake_word_detection()
    
    # Send welcome message if no wake word detection
    if not agent_ctx.has_wake_word_capability():
        session.conversation.item.create(
            llm.ChatMessage(
                role="assistant",
                content=WELCOME_MESSAGE
            )
        )
        session.response.create()
        await agent_ctx.activate_conversation()
    else:
        logger.info("Wake word detection active. Say 'hey computer' or 'hey assistant' to start.")
    
    # Event handlers for speech processing
    @session.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        """Handle committed user speech."""
        if isinstance(msg.content, list):
            msg.content = "\n".join("[image]" if isinstance(x, llm.ChatImage) else x for x in msg.content)
        
        logger.info(f"User said: {msg.content}")
        
        # Cancel any pending deactivation timer - user is actively engaged
        agent_ctx.cancel_deactivation_timer()
        
        # Check if we should respond
        if agent_ctx.should_respond_to_speech():
            handle_user_message(msg)
        else:
            logger.info("Ignoring speech - wake word not detected")
    
    @session.on("agent_speech_committed")  
    def on_agent_speech_committed(msg: llm.ChatMessage):
        """Handle agent speech completion."""
        logger.info(f"Assistant said: {msg.content}")
        
        # For wake word mode, start/restart deactivation timer after agent speaks
        if agent_ctx.has_wake_word_capability():
            agent_ctx.start_deactivation_timer(delay=10)
    

    
    def handle_user_message(msg: llm.ChatMessage):
        """Process user message and generate response."""
        try:
            # Check for specific commands that might need function calls
            content = msg.content.lower() if msg.content else ""
            
            # Handle time/date requests
            if any(word in content for word in ["time", "clock", "hour"]):
                response = agent_ctx.get_current_time()
                send_function_response(response)
            elif any(word in content for word in ["date", "today", "day"]):
                response = agent_ctx.get_current_date()  
                send_function_response(response)
            elif any(word in content for word in ["datetime", "time and date", "date and time"]):
                response = agent_ctx.get_current_datetime()
                send_function_response(response)
            else:
                # Regular conversation
                session.conversation.item.create(
                    llm.ChatMessage(
                        role="user",
                        content=msg.content
                    )
                )
                session.response.create()
                
        except Exception as e:
            logger.error(f"Error handling user message: {e}")
    
    def send_function_response(response: str):
        """Send a function response to the user."""
        session.conversation.item.create(
            llm.ChatMessage(
                role="assistant",
                content=response
            )
        )
        session.response.create()
    
    try:
        # Keep the agent running
        logger.info("Voice agent is running...")
        await asyncio.Event().wait()  # Run indefinitely
        
    except KeyboardInterrupt:
        logger.info("Shutting down voice agent...")
    finally:
        await agent_ctx.stop_wake_word_detection()


def main():
    """Main function to run the voice agent."""
    
    # Validate required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please check your .env file and ensure all required API keys are set.")
        return
    
    # Log configuration
    logger.info("🎙️ Starting LiveKit MultimodalAgent Voice Assistant...")
    logger.info(f"Wake Word Detection: {'✅ Enabled' if os.getenv('PORCUPINE_ACCESS_KEY') else '❌ Disabled (always-on mode)'}")
    logger.info(f"OpenAI Voice: {os.getenv('OPENAI_VOICE', 'alloy')}")
    logger.info(f"Temperature: {os.getenv('OPENAI_TEMPERATURE', '0.8')}")
    
    logger.info("\n📋 Available CLI modes:")
    logger.info("  python livekit.py console  - Terminal mode (local testing)")
    logger.info("  python livekit.py dev      - Development mode (connect to LiveKit)")
    logger.info("  python livekit.py start    - Production mode")
    logger.info("  python livekit.py download-files - Download model files\n")
    
    # Run the agent with LiveKit CLI
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    main()
