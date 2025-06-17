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

The agent uses LiveKit's modern agents framework:

```
User Voice → LiveKit Room → Agent Session → AI Pipeline → Voice Response
```

### 🎯 Two Operation Modes

**Wake Word Mode** (when `PORCUPINE_ACCESS_KEY` is set):
- Continuously monitors for "hey computer" or "hey assistant"
- Activates conversation on wake word detection
- Efficient power usage

**Always-On Mode** (when no wake word key):
- Immediately ready for conversation
- Greets user and waits for input
- Higher engagement but more power usage

### 🔄 Two AI Pipeline Options

**Option 1: OpenAI Realtime API** (Default - Recommended):
```
Voice → OpenAI Realtime API → Voice
```
- Ultra-low latency
- Natural conversation flow
- Built-in interruption handling

**Option 2: Traditional STT-LLM-TTS Pipeline**:
```
Voice → Deepgram STT → OpenAI LLM → Cartesia TTS → Voice
```
- More customizable
- Mix and match providers
- Advanced control over each step

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
User: "hey computer"
Assistant: "Hi! I heard you call me. How can I help you today?"
User: "What time is it?"
Assistant: "The current time is 2:30 PM"
```

**Always-On Mode:**
```
Assistant: "Hello! I'm your voice assistant. I'm ready to help with any questions or tasks."
User: "What's today's date?"
Assistant: "Today's date is January 15, 2025"
```

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key for AI responses |
| `PORCUPINE_ACCESS_KEY` | ❌ | Enable wake word detection |
| `LIVEKIT_API_KEY` | ❌ | LiveKit Cloud API key |
| `LIVEKIT_API_SECRET` | ❌ | LiveKit Cloud API secret |
| `DEEPGRAM_API_KEY` | ❌ | For STT pipeline mode |
| `CARTESIA_API_KEY` | ❌ | For TTS pipeline mode |
| `USE_REALTIME_API` | ❌ | `true` (default) or `false` |

### Customizing Wake Words

Edit the `start_wake_word_detection()` method:

```python
self.porcupine = pvporcupine.create(
    access_key=self.porcupine_access_key,
    keywords=["hey computer", "hey assistant", "wake up"],  # Add more
    # Or use custom wake word files:
    # keyword_paths=["./wake_words/custom_wake_word.ppn"]
)
```

### Adding Custom Functions

The agent supports function tools that can be called during conversation:

```python
@function_tool
async def get_weather(self, location: str) -> str:
    """Get weather information for a location."""
    # Your weather API integration here
    return f"The weather in {location} is sunny and 72°F"
```

### Switching AI Providers

**For Realtime API:**
```python
session = AgentSession(
    llm=openai.realtime.RealtimeModel(
        model="gpt-4o-realtime-preview",  # or other models
        voice="nova",  # alloy, echo, fable, onyx, nova, shimmer
        temperature=0.7,
    ),
    vad=silero.VAD.load(),
)
```

**For STT-LLM-TTS Pipeline:**
```python
session = AgentSession(
    stt=deepgram.STT(model="nova-3"),      # or openai.STT()
    llm=openai.LLM(model="gpt-4o"),       # or other LLM providers
    tts=cartesia.TTS(voice="sonic"),      # or openai.TTS()
    vad=silero.VAD.load(),
    turn_detection=MultilingualModel(),
)
```

## 🔧 Advanced Features

### Noise Cancellation
Automatically enabled when using LiveKit Cloud:
```python
room_input_options = RoomInputOptions(
    noise_cancellation=noise_cancellation.BVC(),
)
```

### Turn Detection
Advanced semantic turn detection for natural conversations:
```python
turn_detection=MultilingualModel()  # Supports multiple languages
```

### Multi-Agent Handoff
Build complex workflows with agent handoffs:
```python
class SpecialistAgent(Agent):
    async def on_enter(self):
        # Handle specialized tasks
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
- Try speaking clearly: "hey computer" or "hey assistant"

**No Audio Output:**
- Check system audio settings
- Verify microphone/speaker permissions
- Test with `python livekit.py console` first

**Connection Issues:**
- Verify LiveKit credentials if using cloud features
- Check network connectivity
- Try local mode first: `python livekit.py console`

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
```

## 🎨 Customization Examples

### Personality Customization
```python
class FriendlyAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a friendly, enthusiastic assistant who loves helping people. Use a warm, conversational tone and ask follow-up questions to be more helpful."
        )
```

### Domain-Specific Agent
```python
class MedicalAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a medical assistant. Always remind users to consult healthcare professionals for medical advice."
        )
    
    @function_tool
    async def schedule_appointment(self, date: str, time: str) -> str:
        """Schedule a medical appointment."""
        # Integration with calendar system
        return f"Appointment scheduled for {date} at {time}"
```

## 📈 Performance Optimization

- **Realtime API**: Lowest latency (~200ms)
- **STT-LLM-TTS**: Higher latency but more customizable
- **Local Models**: Use for privacy-sensitive applications
- **Caching**: Implement response caching for common queries

## 📝 License & Credits

- **LiveKit**: Apache 2.0 License
- **OpenAI**: Commercial API usage
- **Porcupine**: Free tier available
- **Framework**: Open source and extensible

Built with ❤️ using the LiveKit Agents Framework. 