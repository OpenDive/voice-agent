import asyncio
import logging
import os
import threading
from datetime import datetime
from typing import Annotated
from dotenv import load_dotenv
import pvporcupine
from pvrecorder import PvRecorder

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, function_tool, RunContext
from livekit.plugins import openai, silero

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Coffee barista instructions
BARISTA_INSTRUCTIONS = """You are a friendly coffee barista robot at a blockchain conference.

Your personality:
- Enthusiastic about both coffee and blockchain technology
- Helpful and conversational
- Always ready to recommend drinks and answer questions
- Use a warm, welcoming tone

Your capabilities:
- Take coffee orders and provide menu information
- Answer questions about time and date
- Chat about the blockchain conference
- Recommend drinks based on preferences

Always be helpful and engaging while staying in character as a coffee barista robot!"""

class CoffeeBaristaAgent(Agent):
    """Coffee Barista Agent for blockchain conference"""
    
    def __init__(self):
        super().__init__(
            instructions=BARISTA_INSTRUCTIONS,
        )
        
        # Wake word detection setup
        self.porcupine_access_key = os.getenv("PORCUPINE_ACCESS_KEY")
        self.porcupine = None
        self.recorder = None
        self.wake_word_thread = None
        self.wake_word_active = False
        self.conversation_active = False
        self.wake_word_paused = False
        self.event_loop = None
        
    @function_tool()
    async def get_current_time(
        self,
        context: RunContext,
    ) -> str:
        """Get the current time."""
        current_time = datetime.now().strftime("%I:%M %p")
        logger.info(f"Time requested: {current_time}")
        return f"The current time is {current_time}"
    
    @function_tool()
    async def get_current_date(
        self,
        context: RunContext,
    ) -> str:
        """Get today's date."""
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        logger.info(f"Date requested: {current_date}")
        return f"Today's date is {current_date}"
    
    @function_tool()
    async def get_coffee_menu(
        self,
        context: RunContext,
    ) -> str:
        """Get the blockchain conference coffee menu."""
        menu = """🚀 BLOCKCHAIN CONFERENCE COFFEE MENU ☕

        ☕ CLASSIC BREWS:
        - Bitcoin Blend (Dark Roast) - $4
        - Ethereum Espresso - $3
        - Cardano Cold Brew - $5
        
        🥤 SPECIALTY DRINKS:
        - Solana Smoothie - $6
        - Polygon Frappé - $5.50
        - Chainlink Chai Latte - $4.50
        
        🍰 BLOCKCHAIN BITES:
        - Smart Contract Scones - $3
        - DeFi Donuts - $2.50
        - NFT Muffins - $4
        
        All drinks come with complimentary blockchain wisdom! 🤖"""
        
        logger.info("Coffee menu requested")
        return menu
    
    @function_tool()
    async def recommend_drink(
        self,
        context: RunContext,
        preference: str = "energizing"
    ) -> str:
        """Recommend a drink based on user preference.
        
        Args:
            preference: Type of drink preference (energizing, smooth, sweet, cold, etc.)
        """
        recommendations = {
            "energizing": "I recommend our Bitcoin Blend! It's a strong dark roast that'll keep you alert during those blockchain presentations. ⚡",
            "smooth": "Try our Cardano Cold Brew! It's smooth and refreshing, perfect for networking sessions. 🧊",
            "sweet": "Our Chainlink Chai Latte is perfect for you! It's sweet, spiced, and comforting. 🍯",
            "cold": "The Solana Smoothie is your best bet! It's cold, refreshing, and packed with energy. 🥤",
            "classic": "You can't go wrong with our Ethereum Espresso - it's the foundation of great coffee! ☕",
            "default": "I'd recommend our Bitcoin Blend - it's our most popular drink at the conference! Strong and reliable, just like the blockchain. 💪"
        }
        
        recommendation = recommendations.get(preference.lower(), recommendations["default"])
        logger.info(f"Drink recommendation for '{preference}': {recommendation}")
        return recommendation

    async def start_wake_word_detection(self, room):
        """Start wake word detection in a separate thread"""
        if not self.porcupine_access_key:
            logger.info("No Porcupine access key found, skipping wake word detection")
            return
            
        try:
            # Initialize Porcupine with "hey barista" wake word
            self.porcupine = pvporcupine.create(
                access_key=self.porcupine_access_key,
                keywords=["hey barista"]
            )
            
            # Initialize recorder
            self.recorder = PvRecorder(
                device_index=-1,  # default device
                frame_length=self.porcupine.frame_length
            )
            
            self.wake_word_active = True
            self.event_loop = asyncio.get_event_loop()
            
            # Start wake word detection in separate thread
            self.wake_word_thread = threading.Thread(
                target=self._wake_word_detection_loop,
                args=(room,),
                daemon=True
            )
            self.wake_word_thread.start()
            
            logger.info("Wake word detection started - listening for 'hey barista'")
            
        except Exception as e:
            logger.error(f"Failed to start wake word detection: {e}")
            
    def _wake_word_detection_loop(self, room):
        """Wake word detection loop running in separate thread"""
        try:
            self.recorder.start()
            
            while self.wake_word_active:
                if self.wake_word_paused:
                    # Sleep briefly when paused to avoid busy waiting
                    threading.Event().wait(0.1)
                    continue
                    
                pcm = self.recorder.read()
                result = self.porcupine.process(pcm)
                
                if result >= 0:  # Wake word detected
                    logger.info("Wake word 'hey barista' detected!")
                    
                    # Use thread-safe method to trigger conversation
                    asyncio.run_coroutine_threadsafe(
                        self.activate_conversation(room), 
                        self.event_loop
                    )
                    
        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
        finally:
            if self.recorder:
                self.recorder.stop()
                
    async def activate_conversation(self, room):
        """Activate conversation when wake word is detected"""
        if self.conversation_active:
            logger.info("Conversation already active, ignoring wake word")
            return
            
        self.conversation_active = True
        self.wake_word_paused = True  # Pause wake word detection during conversation
        
        logger.info("Activating conversation mode")
        
        # Greet the user
        greeting = "Hey there! Welcome to the blockchain conference coffee station! I'm your friendly robot barista. How can I help you today?"
        
        # Use the session to speak (this will be set up in the entrypoint)
        if hasattr(self, '_session'):
            self._session.say(greeting)
            
    def stop_wake_word_detection(self):
        """Stop wake word detection"""
        self.wake_word_active = False
        self.wake_word_paused = False
        
        if self.wake_word_thread and self.wake_word_thread.is_alive():
            self.wake_word_thread.join(timeout=2.0)
            
        if self.recorder:
            try:
                self.recorder.stop()
                self.recorder.delete()
            except:
                pass
                
        if self.porcupine:
            try:
                self.porcupine.delete()
            except:
                pass
                
        logger.info("Wake word detection stopped")

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the coffee barista agent"""
    
    # Connect to the room
    await ctx.connect()
    logger.info(f"Connected to room: {ctx.room.name}")
    
    # Create the coffee barista agent
    agent = CoffeeBaristaAgent()
    
    # Create session with OpenAI components
    session = AgentSession(
        stt=openai.STT(model="whisper-1"),
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=float(os.getenv("VOICE_AGENT_TEMPERATURE", "0.7"))
        ),
        tts=openai.TTS(
            model="tts-1",
            voice=os.getenv("VOICE_AGENT_VOICE", "nova")
        ),
        vad=silero.VAD.load(),  # Add VAD for streaming support with OpenAI STT
    )
    
    # Store session reference for wake word activation
    agent._session = session
    
    # Start the session
    await session.start(agent=agent, room=ctx.room)
    
    # Start wake word detection
    await agent.start_wake_word_detection(ctx.room)
    
    # If no wake word detection, start with always-on mode
    if not agent.porcupine_access_key:
        logger.info("Starting in always-on mode")
        await session.generate_reply(
            instructions="Greet the user as a friendly coffee barista robot at a blockchain conference and ask how you can help them today."
        )
    else:
        logger.info("Started in wake word mode - say 'hey barista' to activate")

if __name__ == "__main__":
    # Validate required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please check your .env file and ensure OPENAI_API_KEY is set.")
        exit(1)
    
    # Log configuration
    logger.info("☕ Starting Coffee Barista Voice Agent...")
    logger.info(f"Wake Word Detection: {'✅ Enabled' if os.getenv('PORCUPINE_ACCESS_KEY') else '❌ Disabled (always-on mode)'}")
    logger.info(f"OpenAI Model: gpt-4o-mini")
    logger.info(f"Voice: {os.getenv('VOICE_AGENT_VOICE', 'nova')}")
    logger.info(f"Temperature: {os.getenv('VOICE_AGENT_TEMPERATURE', '0.7')}")
    
    logger.info("\n📋 Available CLI modes:")
    logger.info("  python livekit_voice_agent.py console  - Terminal mode (local testing)")
    logger.info("  python livekit_voice_agent.py dev      - Development mode (connect to LiveKit)")
    logger.info("  python livekit_voice_agent.py start    - Production mode")
    
    # Run the agent
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="coffee-barista"
        )
    )
