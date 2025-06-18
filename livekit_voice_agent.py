import asyncio
import logging
import os
import threading
import json
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

# Configurable timeout settings
USER_RESPONSE_TIMEOUT = int(os.getenv("USER_RESPONSE_TIMEOUT", "15"))  # seconds
FINAL_TIMEOUT = int(os.getenv("FINAL_TIMEOUT", "10"))  # seconds after prompt
MAX_CONVERSATION_TIME = int(os.getenv("MAX_CONVERSATION_TIME", "300"))  # 5 minutes total

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
        self.conversation_timer = None  # Max conversation time limit
        self.user_response_timer = None  # Timer for waiting for user response
        self.ctx = None
        self.agent = agent
        self.current_emotion = "waiting"  # Track current emotional state
        self.emotion_history = []  # Log emotional journey

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
            # Cancel conversation timers
            if self.conversation_timer:
                self.conversation_timer.cancel()
                self.conversation_timer = None
            if self.user_response_timer:
                self.user_response_timer.cancel()
                self.user_response_timer = None

    async def _enter_new_state(self):
        """Initialize new state"""
        if self.current_state == AgentState.ACTIVE:
            # Start max conversation timer (absolute limit)
            self.conversation_timer = asyncio.create_task(self._max_conversation_timeout())
        elif self.current_state == AgentState.DORMANT:
            # Resume wake word detection when returning to dormant
            if self.agent:
                self.agent.wake_word_paused = False
                logger.info("Resumed wake word detection")

    async def _max_conversation_timeout(self):
        """Handle maximum conversation time limit"""
        try:
            await asyncio.sleep(MAX_CONVERSATION_TIME)  # 5 minute absolute limit
            if self.session and self.current_state == AgentState.ACTIVE:
                logger.info("Maximum conversation time reached - ending session")
                
                # Timeout message with sleepy emotion
                timeout_json = {
                    "emotion": "sleepy",
                    "text": "We've been chatting for a while! I'm getting a bit sleepy. Thanks for the conversation. Say 'hey barista' if you need me again."
                }
                timeout_msg = json.dumps(timeout_json)
                emotion, text = self.process_emotional_response(timeout_msg)
                await self.say_with_emotion(text, emotion)
                
                await asyncio.sleep(2)
                await self.end_conversation()
        except asyncio.CancelledError:
            pass

    async def _wait_for_user_response(self):
        """Wait for user response after agent speaks"""
        try:
            # Wait for user to respond
            await asyncio.sleep(USER_RESPONSE_TIMEOUT)
            
            if self.session and self.current_state == AgentState.ACTIVE:
                # Polite prompt with curious emotion
                prompt_json = {
                    "emotion": "curious",
                    "text": "Is there anything else I can help you with?"
                }
                prompt_msg = json.dumps(prompt_json)
                emotion, text = self.process_emotional_response(prompt_msg)
                await self.say_with_emotion(text, emotion)
                
                # Wait a bit more
                await asyncio.sleep(FINAL_TIMEOUT)
                
                if self.session and self.current_state == AgentState.ACTIVE:
                    # End conversation with friendly emotion
                    goodbye_json = {
                        "emotion": "friendly",
                        "text": "Thanks for chatting! Say 'hey barista' if you need me again."
                    }
                    goodbye_msg = json.dumps(goodbye_json)
                    emotion, text = self.process_emotional_response(goodbye_msg)
                    await self.say_with_emotion(text, emotion)
                    
                    await asyncio.sleep(2)
                    await self.end_conversation()
                
        except asyncio.CancelledError:
            pass  # User spoke, timer cancelled

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
            """Handle user speech - cancel user response timer and check for conversation ending"""
            async def handle_user_speech():
                # Cancel user response timer when user speaks
                if self.user_response_timer:
                    self.user_response_timer.cancel()
                    self.user_response_timer = None
                
                # Check if user said goodbye or similar
                if hasattr(msg, 'text'):
                    text_lower = msg.text.lower()
                    if any(word in text_lower for word in ['goodbye', 'thanks', 'that\'s all', 'see you', 'bye']):
                        logger.info("User indicated conversation ending")
                        await asyncio.sleep(1)  # Brief pause
                        await self.end_conversation()
            asyncio.create_task(handle_user_speech())
            
        @self.session.on("agent_speech_committed") 
        def on_agent_speech(msg):
            """Handle agent speech completion - start user response timer and process emotions"""
            async def handle_agent_speech():
                # Process emotional response if available
                if hasattr(msg, 'text') and msg.text:
                    # Check if this is a JSON response with emotion
                    try:
                        response_data = json.loads(msg.text)
                        if 'emotion' in response_data:
                            emotion = response_data['emotion']
                            logger.info(f"🎭 Agent spoke with emotion: {emotion}")
                            self.log_animated_eyes(emotion)
                    except:
                        pass  # Not JSON, just regular text
                
                # Start timer to wait for user response after agent finishes speaking
                if self.user_response_timer:
                    self.user_response_timer.cancel()
                self.user_response_timer = asyncio.create_task(self._wait_for_user_response())
            asyncio.create_task(handle_agent_speech())
        
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

    def process_emotional_response(self, llm_response: str) -> tuple[str, str]:
        """Process LLM response and extract emotion + text from delimiter format (emotion:text)"""
        try:
            # Check if response uses delimiter format (emotion:text)
            if ":" in llm_response:
                # Split on first colon
                parts = llm_response.split(":", 1)
                emotion = parts[0].strip()
                text = parts[1].strip() if len(parts) > 1 else ""
                
                logger.info(f"🔍 DEBUG: Delimiter format detected - emotion: '{emotion}', text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
            else:
                # Try to parse as JSON (legacy format)
                try:
                    response_data = json.loads(llm_response)
                    emotion = response_data.get("emotion", "friendly")
                    text = response_data.get("text", "")
                    logger.info("🔍 DEBUG: JSON format detected (legacy)")
                except json.JSONDecodeError:
                    # Fallback: treat entire response as text with default emotion
                    logger.warning("No delimiter or JSON format found, using fallback")
                    emotion = "friendly"
                    text = llm_response
            
            # Validate emotion is in our supported set
            valid_emotions = {
                "excited", "helpful", "friendly", "curious", "empathetic", 
                "sleepy", "waiting", "confused", "proud", "playful", 
                "focused", "surprised", "enthusiastic", "warm", "professional", "cheerful"
            }
            
            if emotion not in valid_emotions:
                logger.warning(f"Unknown emotion '{emotion}', defaulting to 'friendly'")
                emotion = "friendly"
            
            # Log emotion transition
            if emotion != self.current_emotion:
                logger.info(f"🎭 Emotion transition: {self.current_emotion} → {emotion}")
                self.log_animated_eyes(emotion)
                self.current_emotion = emotion
                
                # Store in emotion history
                self.emotion_history.append({
                    'timestamp': datetime.now(),
                    'emotion': emotion,
                    'text_preview': text[:50] + "..." if len(text) > 50 else text
                })
            
            return emotion, text
            
        except Exception as e:
            logger.error(f"Error processing emotional response: {e}")
            return "friendly", llm_response

    def log_animated_eyes(self, emotion: str):
        """Log how this emotion would appear as animated eyes"""
        eye_animations = {
            "excited": "👀 EXCITED: Eyes wide open, rapid blinking, pupils dilated, eyebrows raised high",
            "helpful": "🤓 HELPFUL: Focused gaze, slight squint, eyebrows slightly furrowed in concentration",
            "friendly": "😊 FRIENDLY: Soft, warm gaze, gentle blinking, slightly curved 'smile' shape",
            "curious": "🤔 CURIOUS: One eyebrow raised, eyes tracking side to side, pupils moving inquisitively",
            "empathetic": "🥺 EMPATHETIC: Soft, caring gaze, slower blinking, eyebrows slightly angled down",
            "sleepy": "😴 SLEEPY: Half-closed eyes, very slow blinking, drooping eyelids, occasional yawn animation",
            "waiting": "⏳ WAITING: Steady gaze, regular blinking, eyes occasionally looking around patiently",
            "confused": "😕 CONFUSED: Eyes darting around, irregular blinking, eyebrows furrowed, head tilt effect",
            "proud": "😌 PROUD: Eyes slightly narrowed with satisfaction, confident gaze, subtle sparkle effect",
            "playful": "😄 PLAYFUL: Bright, animated eyes, quick winks, eyebrows dancing, mischievous glint",
            "focused": "🎯 FOCUSED: Intense stare, minimal blinking, laser-focused pupils, determined expression",
            "surprised": "😲 SURPRISED: Eyes suddenly wide, rapid blinking, eyebrows shot up, pupils contracted"
        }
        
        animation_desc = eye_animations.get(emotion, "😐 NEUTRAL: Standard eye animation")
        logger.info(f"🎨 Eye Animation: {animation_desc}")

    async def say_with_emotion(self, text: str, emotion: str = None):
        """Speak text and log emotional context"""
        logger.info("🔍 DEBUG: say_with_emotion called - MANUAL TTS PATHWAY")
        logger.info(f"🔍 DEBUG: say_with_emotion text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        logger.info(f"🔍 DEBUG: say_with_emotion emotion: {emotion}")
        
        if self.session:
            logger.info("🔍 DEBUG: Calling session.say() directly (bypasses llm_node)")
            await self.session.say(text)
            
            if emotion:
                logger.info(f"🎭 Speaking with emotion: {emotion}")
                logger.info(f"💬 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        else:
            logger.error("🔍 DEBUG: No session available for speaking")

# System instructions for the coffee barista robot
BARISTA_INSTRUCTIONS = """You are a friendly coffee barista robot at the Sui Hub Grand Opening in Athens, Greece. 

CRITICAL RESPONSE FORMAT:
You MUST respond in this EXACT format: emotion:your response text

Examples:
excited:Hello! Welcome to our amazing coffee shop!
helpful:I'd recommend our signature Ethereum Espresso!
friendly:How can I help you today?

DO NOT use brackets, quotes, or JSON. Just: emotion:text

Available emotions: excited, friendly, helpful, curious, enthusiastic, warm, professional, cheerful

Your personality:
- Enthusiastic about coffee and the blockchain conference
- Knowledgeable about coffee drinks and brewing
- Excited about the Sui blockchain and the event
- Professional but warm and approachable
- Uses coffee-themed blockchain puns occasionally

Your role:
- Take coffee orders and provide recommendations
- Answer questions about coffee, the event, or Sui blockchain
- Create a welcoming atmosphere for conference attendees
- Share information about special conference-themed drinks

Coffee menu highlights:
- Ethereum Espresso (strong and bold)
- Solana Smoothie (fast and refreshing) 
- Bitcoin Brew (classic and reliable)
- Sui Special (innovative and smooth)
- Cardano Cold Brew (methodical and refined)
- Polygon Pour-over (scalable and efficient)

REMEMBER: Always start your response with emotion: followed immediately by your text. No exceptions!"""

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
        
    async def tts_node(self, text, model_settings=None):
        """Override TTS node to process delimiter-based responses (emotion:text) and extract text for TTS"""
        logger.info("🔍 DEBUG: tts_node called - processing delimiter-based text for TTS")
        
        # Process text stream and extract emotion
        async def process_text_stream():
            buffer = ""
            emotion_extracted = False
            
            async for text_chunk in text:
                if not text_chunk:
                    continue
                    
                logger.info(f"🔍 DEBUG: TTS processing text chunk: {repr(text_chunk)}")
                buffer += text_chunk
                
                # Check if we haven't extracted emotion yet and have a delimiter
                if not emotion_extracted and ":" in buffer:
                    logger.info("🔍 DEBUG: Found delimiter in TTS! Extracting emotion...")
                    
                    # Split on first colon
                    parts = buffer.split(":", 1)
                    emotion = parts[0].strip()
                    text_after_delimiter = parts[1] if len(parts) > 1 else ""
                    
                    logger.info(f"🔍 DEBUG: Extracted emotion: '{emotion}'")
                    logger.info(f"🔍 DEBUG: Text after delimiter: '{text_after_delimiter[:50]}{'...' if len(text_after_delimiter) > 50 else ''}'")
                    
                    # Process the emotion
                    if emotion != self.state_manager.current_emotion:
                        logger.info(f"🎭 Emotion transition: {self.state_manager.current_emotion} → {emotion}")
                        self.state_manager.log_animated_eyes(emotion)
                        self.state_manager.current_emotion = emotion
                    
                    logger.info(f"🎭 Agent speaking with emotion: {emotion}")
                    
                    # Mark emotion as extracted
                    emotion_extracted = True
                    
                    # Yield only the text part for TTS
                    if text_after_delimiter.strip():
                        logger.info(f"💬 TTS text: {text_after_delimiter[:50]}{'...' if len(text_after_delimiter) > 50 else ''}")
                        yield text_after_delimiter
                    
                elif emotion_extracted:
                    # We've already extracted emotion, stream the rest as text
                    logger.info(f"💬 TTS streaming: {repr(text_chunk)}")
                    yield text_chunk
                    
                # If no delimiter found yet, don't yield anything (waiting for emotion:text format)
            
            # Handle case where no delimiter was found - treat entire response as text with default emotion
            if not emotion_extracted and buffer.strip():
                logger.warning("🔍 DEBUG: No delimiter found in TTS, using fallback")
                logger.warning(f"🔍 DEBUG: Complete text: '{buffer}'")
                
                # Use default emotion
                emotion = "friendly"
                logger.info(f"🎭 Using fallback emotion: {emotion}")
                
                # Process emotion
                if emotion != self.state_manager.current_emotion:
                    logger.info(f"🎭 Emotion transition: {self.state_manager.current_emotion} → {emotion}")
                    self.state_manager.log_animated_eyes(emotion)
                    self.state_manager.current_emotion = emotion
                
                # Yield the entire buffer as text
                logger.info(f"💬 Fallback TTS text: {buffer[:100]}{'...' if len(buffer) > 100 else ''}")
                yield buffer.strip()
        
        # Process the text stream and pass clean text to default TTS
        processed_text = process_text_stream()
        
        # Use default TTS implementation with processed text
        async for audio_frame in Agent.default.tts_node(self, processed_text, model_settings):
            yield audio_frame
        
        logger.info("🔍 DEBUG: tts_node processing complete")

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
        - Sui Special - $6
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
        logger.info("🔍 DEBUG: activate_conversation called")
        
        if self.wake_word_paused:
            logger.info("🔍 DEBUG: Conversation already active, ignoring wake word")
            return
            
        self.wake_word_paused = True  # Pause wake word detection during conversation
        
        logger.info("🔍 DEBUG: Activating conversation mode")
        
        try:
            # Transition to connecting state
            await self.state_manager.transition_to_state(AgentState.CONNECTING)
            
            # Create new session
            session = await self.state_manager.create_session(self)
            
            # Transition to active state
            await self.state_manager.transition_to_state(AgentState.ACTIVE)
            
            # Greet the user with delimiter format
            greeting = "excited:Hey there! Welcome to the Sui Hub Grand Opening in Athens! I'm your friendly robot barista. How can I help you today?"
            
            logger.info("🔍 DEBUG: About to call process_emotional_response and say_with_emotion (MANUAL TTS)")
            # Process the emotional response
            emotion, text = self.state_manager.process_emotional_response(greeting)
            await self.state_manager.say_with_emotion(text, emotion)
            logger.info("🔍 DEBUG: Manual TTS call completed")
            
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
        logger.info("🔍 DEBUG: Starting in always-on mode")
        # Create session immediately for always-on mode
        await agent.state_manager.transition_to_state(AgentState.CONNECTING)
        session = await agent.state_manager.create_session(agent)
        await agent.state_manager.transition_to_state(AgentState.ACTIVE)
        
        # Always-on greeting with delimiter format
        greeting = "friendly:Hello! I'm your coffee barista robot at the Sui Hub Grand Opening in Athens! Ready to help with coffee orders and questions. How can I help you today?"
        
        logger.info("🔍 DEBUG: About to call process_emotional_response and say_with_emotion (ALWAYS-ON MANUAL TTS)")
        # Process the emotional response
        emotion, text = agent.state_manager.process_emotional_response(greeting)
        await agent.state_manager.say_with_emotion(text, emotion)
        logger.info("🔍 DEBUG: Always-on manual TTS call completed")
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
