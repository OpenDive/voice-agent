# LiveKit Coffee Barista Voice Agent

This is a production-ready Python voice agent built using the **LiveKit Agents Framework** with OpenAI Realtime API integration. The agent acts as a helpful coffee barista robot at a blockchain conference, supporting wake word detection for hands-free interaction.

## 🎯 Features

- **🎙️ MultimodalAgent Architecture**: Built with LiveKit's advanced MultimodalAgent framework
- **🔊 Smart Wake Word Detection**: "Hey Barista" activation with intelligent conversation management
- **🤖 OpenAI Realtime API**: Ultra-low latency voice-to-voice interaction (~200ms)
- **⚡ Thread-Safe State Management**: Robust multi-threaded wake word detection
- **🔄 Smart Timer Management**: Automatic conversation timeout with user engagement detection
- **🛡️ Duplicate Protection**: Prevents multiple wake word activations during conversation
- **☕ Coffee Barista Theme**: Specialized for coffee ordering and blockchain conference context
- **🛠️ Function Tools**: Built-in time/date functions with easy extensibility
- **📞 Multi-mode Support**: Terminal, development, and production modes
- **🎵 Audio Processing**: Advanced VAD, turn detection, and conversation flow

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Required: OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Wake Word Detection
PORCUPINE_ACCESS_KEY=your_porcupine_access_key_here

# Optional: LiveKit Cloud (recommended for production)
LIVEKIT_API_KEY=your_livekit_api_key_here
LIVEKIT_API_SECRET=your_livekit_api_secret_here
LIVEKIT_URL=wss://your-livekit-server.com

# Agent Configuration
VOICE_AGENT_TEMPERATURE=0.7  # AI response creativity (0.0-1.0)
VOICE_AGENT_VOICE=nova      # OpenAI voice: alloy, echo, fable, onyx, nova, shimmer
```

### 3. Get API Keys

#### 🔑 OpenAI API Key (Required)
1. Sign up at [OpenAI Platform](https://platform.openai.com/)
2. Create an API key
3. Add to your `.env` file

#### 🎙️ Porcupine Access Key (Optional - for wake words)
1. Sign up at [Picovoice Console](https://console.picovoice.ai/)
2. Create a project and get your Access Key
3. Add to your `.env` file

#### ☁️ LiveKit Credentials (Optional - for cloud features)
1. Sign up at [LiveKit Cloud](https://livekit.io/)
2. Create a project
3. Copy API Key and Secret

### 4. Download Model Files

```bash
python livekit.py download-files
```

### 5. Run the Agent

```bash
# Terminal mode (local testing)
python livekit.py console

# Development mode (connect to LiveKit)
python livekit.py dev

# Production mode
python livekit.py start
```

## 🏗️ Architecture

The coffee barista agent uses LiveKit's advanced MultimodalAgent framework with thread-safe wake word detection:

```
User Voice → LiveKit Room → MultimodalAgent Session → OpenAI Realtime API → Voice Response
                              ↑
                    Wake Word Detection Thread
```

### 🎯 Operational Modes

**Wake Word Mode** (when `PORCUPINE_ACCESS_KEY` is set):
- Continuously monitors for "hey barista"
- Activates conversation on wake word detection
- Intelligent conversation state management
- Automatic wake word pausing during conversation
- Smart timer-based conversation timeout

**Always-On Mode** (when no wake word key):
- Immediately ready for conversation
- Greets user as coffee barista
- Higher engagement but more power usage

### 🚀 Key Technical Improvements

**Thread-Safe Wake Word Detection**:
- Uses `asyncio.run_coroutine_threadsafe()` for thread safety
- Prevents race conditions between wake word and main threads
- Duplicate activation protection

**Smart Timer Management**:
- Single timer tracking (prevents multiple concurrent timers)
- Automatic cancellation when user speaks
- Intelligent restart after agent responses
- Proper cleanup on conversation end

**Conversation State Management**:
- Pauses wake word detection during active conversation
- Prevents multiple simultaneous activations
- Graceful conversation ending with timer-based timeout

## 📖 Usage Examples

### Terminal Mode
```bash
python livekit.py console
```
Perfect for development and testing. Speaks directly through your computer's speakers.

### Development Mode
```bash
python livekit.py dev
```
Connects to LiveKit server. Access via:
- [Agents Playground](https://agents-playground.livekit.io/)
- Custom web/mobile apps
- Phone calls (with SIP integration)

### Example Conversation

**Wake Word Mode:**
```
User: "hey barista"
Barista: "Hey there! Welcome to the blockchain conference coffee station! I'm your friendly robot barista. How can I help you today?"
User: "What time is it?"
Barista: "The current time is 2:30 PM. Perfect time for an afternoon coffee! Would you like me to recommend something?"
```

**Always-On Mode:**
```
Barista: "Hello! I'm your robot barista here at the blockchain conference! Ready to help with coffee orders, questions about the event, or just chat!"
User: "What's today's date?"
Barista: "Today's date is January 15, 2025. Great day for the conference! Can I get you something to drink?"
```

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key for Realtime API |
| `PORCUPINE_ACCESS_KEY` | ❌ | Enable "hey barista" wake word detection |
| `LIVEKIT_API_KEY` | ❌ | LiveKit Cloud API key (recommended) |
| `LIVEKIT_API_SECRET` | ❌ | LiveKit Cloud API secret |
| `VOICE_AGENT_TEMPERATURE` | ❌ | AI creativity level (0.0-1.0, default: 0.7) |
| `VOICE_AGENT_VOICE` | ❌ | OpenAI voice (default: nova) |

### Customizing Wake Words

The agent currently uses "hey barista" as the wake word. To customize, edit the `start_wake_word_detection()` method:

```python
self.porcupine = pvporcupine.create(
    access_key=self.porcupine_access_key,
    keywords=["hey barista", "coffee bot", "hey coffee"],  # Add more keywords
    # Or use custom wake word files:
    # keyword_paths=["./wake_words/custom_wake_word.ppn"]
)
```

Note: The agent will automatically pause wake word detection during active conversations to prevent duplicate activations.

### Adding Custom Functions

The agent supports function tools that can be called during conversation:

```python
@function_tool
async def get_weather(self, location: str) -> str:
    """Get weather information for a location."""
    # Your weather API integration here
    return f"The weather in {location} is sunny and 72°F"
```

### Customizing the AI Model

The agent uses OpenAI's Realtime API with the MultimodalAgent architecture:

```python
# In the agent configuration
model = openai.realtime.RealtimeModel(
    model="gpt-4o-realtime-preview",
    voice=os.getenv("VOICE_AGENT_VOICE", "nova"),
    temperature=float(os.getenv("VOICE_AGENT_TEMPERATURE", "0.7")),
    instructions="""You are a helpful robot barista at a blockchain conference..."""
)

agent = MultimodalAgent(model=model)
```

**Available Voices**: alloy, echo, fable, onyx, nova, shimmer
**Temperature Range**: 0.0 (deterministic) to 1.0 (creative)

## 🔧 Advanced Features

### Thread-Safe Wake Word Management
The agent implements sophisticated thread safety:
```python
# Safe state transitions between threads
asyncio.run_coroutine_threadsafe(
    self.activate_conversation(room), 
    self.event_loop
)

# Automatic wake word pausing during conversation
self.wake_word_paused = True  # Prevents duplicate activations
```

### Smart Timer Management
Intelligent conversation timeout with user engagement detection:
```python
# Single timer tracking prevents multiple concurrent timers
if self.timeout_timer:
    self.timeout_timer.cancel()

# Restarts after agent speaks, cancelled when user speaks
self.timeout_timer = asyncio.create_task(self.conversation_timeout())
```

### Event-Driven Architecture
Direct session management with real-time event handling:
```python
@session.on("user_speech_committed")
async def on_user_speech_committed(self, message: rtc.ChatMessage):
    # Process user speech and manage conversation flow
    pass
```

## 📞 Integration Options

### Web & Mobile Apps
Use LiveKit's client SDKs:
- **JavaScript/React**: `livekit-client`
- **iOS/macOS**: `client-sdk-swift`
- **Android**: `client-sdk-android`
- **Flutter**: `client-sdk-flutter`

### Telephony
Connect to phone systems using LiveKit SIP:
```bash
# Your agent can receive phone calls!
```

### WebRTC
Direct browser integration without downloads or apps.

## 🚀 Deployment

### Development
```bash
python livekit.py dev
```

### Production
```bash
python livekit.py start
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "livekit.py", "start"]
```

### Kubernetes
LiveKit agents support horizontal scaling and load balancing.

## 🛠️ Troubleshooting

### Common Issues

**Wake Word Not Working:**
- Check `PORCUPINE_ACCESS_KEY` is set correctly
- Ensure microphone permissions are granted
- Try speaking clearly: "hey barista"
- Check that wake word detection isn't paused during conversation

**No Audio Output:**
- Check system audio settings
- Verify microphone/speaker permissions
- Test with `python livekit.py console` first

**Connection Issues:**
- Verify LiveKit credentials if using cloud features
- Check network connectivity
- Try local mode first: `python livekit.py console`

**Multiple Wake Word Activations:**
- Agent automatically prevents duplicate activations
- Wake word detection pauses during active conversation
- If issues persist, check for race conditions in logs

**Timer Issues:**
- Agent uses single timer tracking to prevent conflicts
- Timer cancels when user speaks, restarts after agent responds
- Check logs for timer management debug information

**Thread Safety Issues:**
- All state changes use `asyncio.run_coroutine_threadsafe()`
- Wake word detection runs in separate thread
- If experiencing crashes, check for proper event loop handling

### Debug Mode
```python
logging.basicConfig(level=logging.DEBUG)
```

### Testing Components
```bash
# Test microphone
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test wake word detection
python -c "import pvporcupine; print('Porcupine available')"

# Test OpenAI connection
python -c "from openai import OpenAI; print('OpenAI available')"

# Test thread safety
python -c "import asyncio; print('Event loop support available')"
```

## 🎨 Customization Examples

### Coffee Shop Personality
The current implementation uses a coffee barista theme:
```python
instructions = """You are a helpful robot barista at a blockchain conference. 
You're enthusiastic about coffee and technology. Keep responses conversational 
and offer coffee recommendations when appropriate."""
```

### Wake Word Customization
```python
# Multiple wake words for coffee context
self.porcupine = pvporcupine.create(
    access_key=self.porcupine_access_key,
    keywords=["hey barista", "coffee bot", "hey coffee"]
)
```

### Custom Function Tools
```python
@function_tool
async def get_coffee_menu(self) -> str:
    """Get the current coffee menu."""
    return "Today's specials: Blockchain Blend, Crypto Cappuccino, DeFi Decaf"

@function_tool  
async def place_order(self, drink: str, size: str) -> str:
    """Place a coffee order."""
    return f"Perfect! I've noted your order for a {size} {drink}. It'll be ready shortly!"
```

## 🔄 Recent Improvements & Fixes

### Version History

**Latest Update: Production-Ready Voice Agent**
- ✅ **Timer Race Conditions Fixed**: Implemented single timer tracking with proper cancellation
- ✅ **Thread Safety Resolved**: Added `asyncio.run_coroutine_threadsafe()` for safe state transitions  
- ✅ **Multiple Wake Word Protection**: Prevents duplicate activations during conversation
- ✅ **Smart Conversation Management**: Wake word detection pauses during active conversation
- ✅ **Event-Driven Architecture**: Direct session management with real-time event handling
- ✅ **Coffee Barista Theme**: Specialized for blockchain conference coffee station

**Key Technical Improvements**:
- Upgraded from basic Agent to MultimodalAgent architecture
- Implemented sophisticated timer management system
- Added comprehensive thread safety measures
- Built intelligent conversation state tracking
- Enhanced wake word detection with pause/resume capability

**Performance Characteristics**:
- ⚡ **Ultra-Low Latency**: ~200ms with OpenAI Realtime API
- 🛡️ **Thread-Safe**: Robust multi-threaded wake word detection
- 🎯 **Smart Activation**: Intelligent wake word activation without duplicates
- ⏱️ **Automatic Timeout**: Graceful conversation ending with timer management
- 🔄 **State Management**: Clean conversation lifecycle with proper cleanup

## 📈 Performance Optimization

- **OpenAI Realtime API**: Lowest latency (~200ms) with built-in turn detection
- **MultimodalAgent**: Production-grade architecture with advanced session management
- **Thread Safety**: Prevents race conditions and crashes in multi-threaded environment
- **Smart Timers**: Single timer tracking eliminates multiple concurrent timer conflicts

## 📝 License & Credits

- **LiveKit**: Apache 2.0 License
- **OpenAI**: Commercial API usage
- **Porcupine**: Free tier available
- **Framework**: Open source and extensible

Built with ❤️ using the LiveKit Agents Framework. 