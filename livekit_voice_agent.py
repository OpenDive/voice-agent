import asyncio
import logging
import os
import threading
from datetime import datetime
from typing import Annotated
from enum import Enum
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

class AgentState(Enum):
    DORMANT = "dormant"           # Only wake word detection active
    CONNECTING = "connecting"     # Creating LiveKit session  
    ACTIVE = "active"            # Conversation mode
    SPEAKING = "speaking"        # Agent is speaking
    DISCONNECTING = "disconnecting" # Cleaning up session

class StateManager:
    def __init__(self, agent=None):
        self.current_state = AgentState.DORMANT
        self.state_lock = asyncio.Lock()
        self.session = None
        self.conversation_timer = None
        self.ctx = None
        self.agent = agent

    async def transition_to_state(self, new_state: AgentState):
        """Handle state transitions with proper cleanup"""
        async with self.state_lock:
            if self.current_state == new_state:
                return
                
            logger.info(f"State transition: {self.current_state.value} → {new_state.value}")
            await self._exit_current_state()
            self.current_state = new_state
            await self._enter_new_state()

    async def _exit_current_state(self):
        """Clean up current state"""
        if self.current_state == AgentState.ACTIVE:
            # Cancel conversation timer
            if self.conversation_timer:
                self.conversation_timer.cancel()
                self.conversation_timer = None

    async def _enter_new_state(self):
        """Initialize new state"""
        if self.current_state == AgentState.ACTIVE:
            # Start conversation timer
            self.conversation_timer = asyncio.create_task(self._conversation_timeout())
        elif self.current_state == AgentState.DORMANT:
            # Resume wake word detection when returning to dormant
            if self.agent:
                self.agent.wake_word_paused = False
                logger.info("Resumed wake word detection")

    async def _conversation_timeout(self):
        """Handle conversation timeout"""
        try:
            # First timeout - prompt user (10 seconds)
            await asyncio.sleep(10)
            if self.session and self.current_state == AgentState.ACTIVE:
                await self.session.say("Are you still there? Is there anything else I can help you with?")
                
                # Second timeout - end conversation (20 more seconds)
                await asyncio.sleep(20)
                if self.session and self.current_state == AgentState.ACTIVE:
                    logger.info("Conversation timeout - ending session")
                    await self.session.say("Thanks for chatting! Say 'hey barista' if you need me again.")
                    await asyncio.sleep(2)  # Let the goodbye message play
                    await self.end_conversation()
        except asyncio.CancelledError:
            pass

    async def create_session(self, agent) -> AgentSession:
        """Create new session when wake word detected"""
        if self.session:
            await self.destroy_session()
            
        self.session = AgentSession(
            stt=openai.STT(model="whisper-1"),
            llm=openai.LLM(
                model="gpt-4o-mini",
                temperature=float(os.getenv("VOICE_AGENT_TEMPERATURE", "0.7"))
            ),
            tts=openai.TTS(
                model="tts-1",
                voice=os.getenv("VOICE_AGENT_VOICE", "nova")
            ),
            vad=silero.VAD.load(),
        )
        
        # Set up session event handlers
        @self.session.on("user_speech_committed")
        def on_user_speech(msg):
            """Reset conversation timer when user speaks"""
            async def reset_timer():
                if self.conversation_timer:
                    self.conversation_timer.cancel()
                self.conversation_timer = asyncio.create_task(self._conversation_timeout())
            asyncio.create_task(reset_timer())
            
        @self.session.on("agent_speech_committed") 
        def on_agent_speech(msg):
            """Handle agent speech completion"""
            async def check_ending():
                # Check if user said goodbye or similar
                if hasattr(msg, 'text'):
                    text_lower = msg.text.lower()
                    if any(word in text_lower for word in ['goodbye', 'thanks', 'that\'s all', 'see you']):
                        logger.info("Detected conversation ending phrase")
                        await asyncio.sleep(2)  # Brief pause before ending
                        await self.end_conversation()
            asyncio.create_task(check_ending())
        
        await self.session.start(agent=agent, room=self.ctx.room)
        return self.session

    async def destroy_session(self):
        """Clean up session when conversation ends"""
        if self.session:
            try:
                await self.session.aclose()
            except Exception as e:
                logger.error(f"Error closing session: {e}")
            finally:
                self.session = None

    async def end_conversation(self):
        """End the current conversation and return to dormant state"""
        await self.destroy_session()
        await self.transition_to_state(AgentState.DORMANT)

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
        
        # State management
        self.state_manager = StateManager(self)
        
        # Wake word detection setup
        self.porcupine_access_key = os.getenv("PORCUPINE_ACCESS_KEY")
        self.porcupine = None
        self.recorder = None
        self.wake_word_thread = None
        self.wake_word_active = False
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
        if self.wake_word_paused:
            logger.info("Conversation already active, ignoring wake word")
            return
            
        self.wake_word_paused = True  # Pause wake word detection during conversation
        
        logger.info("Activating conversation mode")
        
        try:
            # Transition to connecting state
            await self.state_manager.transition_to_state(AgentState.CONNECTING)
            
            # Create new session
            session = await self.state_manager.create_session(self)
            
            # Transition to active state
            await self.state_manager.transition_to_state(AgentState.ACTIVE)
            
            # Greet the user
            greeting = "Hey there! Welcome to the blockchain conference coffee station! I'm your friendly robot barista. How can I help you today?"
            await session.say(greeting)
            
        except Exception as e:
            logger.error(f"Error activating conversation: {e}")
            # Return to dormant state on error
            await self.state_manager.transition_to_state(AgentState.DORMANT)
            self.wake_word_paused = False

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
    
    # Set context in state manager
    agent.state_manager.ctx = ctx
    
    # Start wake word detection
    await agent.start_wake_word_detection(ctx.room)
    
    # If no wake word detection, start with always-on mode
    if not agent.porcupine_access_key:
        logger.info("Starting in always-on mode")
        # Create session immediately for always-on mode
        await agent.state_manager.transition_to_state(AgentState.CONNECTING)
        session = await agent.state_manager.create_session(agent)
        await agent.state_manager.transition_to_state(AgentState.ACTIVE)
        
        await session.say("Hello! I'm your coffee barista robot at the blockchain conference! Ready to help with coffee orders and questions. How can I help you today?")
    else:
        logger.info("Started in wake word mode - say 'hey barista' to activate")
        # Stay in dormant state, waiting for wake word

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
