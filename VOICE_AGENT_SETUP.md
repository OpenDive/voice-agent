# LiveKit Voice Agent with OpenAI Integration

This is a comprehensive Python voice agent that integrates LiveKit's audio processing capabilities with OpenAI's conversational AI. The agent uses wake word detection and provides full voice-to-voice interaction.

## Features

- **Wake Word Detection**: Uses Porcupine for hands-free activation
- **Speech-to-Text**: OpenAI Whisper for accurate transcription
- **Conversational AI**: OpenAI GPT for intelligent responses
- **Text-to-Speech**: OpenAI TTS for natural voice synthesis
- **LiveKit Integration**: Real-time audio processing and room management
- **Multi-threaded Design**: Efficient audio processing and conversation handling

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Required: Porcupine Wake Word Detection
PORCUPINE_ACCESS_KEY=your_porcupine_access_key_here

# Required: OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: LiveKit Configuration (for cloud deployment)
LIVEKIT_API_KEY=your_livekit_api_key_here
LIVEKIT_API_SECRET=your_livekit_api_secret_here
LIVEKIT_URL=wss://your-livekit-server.com
```

### 3. Get API Keys

#### Porcupine Access Key
1. Sign up at [Picovoice Console](https://console.picovoice.ai/)
2. Create a new project
3. Copy your Access Key

#### OpenAI API Key
1. Sign up at [OpenAI](https://platform.openai.com/)
2. Go to API Keys section
3. Create a new API key
4. Copy the key

#### LiveKit Credentials (Optional)
1. Sign up at [LiveKit Cloud](https://livekit.io/)
2. Create a new project
3. Copy your API Key and Secret

## Usage

### Running the Voice Agent

```bash
python livekit.py
```

### Voice Interaction Flow

1. **Wake Word**: Say "hey computer" or "hey assistant"
2. **Listen**: The agent will indicate it's listening
3. **Speak**: Say your question or command (you have 5 seconds)
4. **Response**: The agent will respond with voice

### Example Conversation

```
User: "hey computer"
Agent: "Wake word detected! Starting conversation... Listening for your question..."
User: "What's the weather like today?"
Agent: "I'm a voice assistant, but I don't have access to real-time weather data..."
```

## Architecture

### Components

1. **LiveKitVoiceAgent**: Main class that orchestrates all components
2. **Wake Word Detection**: Continuous monitoring using Porcupine
3. **Audio Recording**: Captures conversation audio using sounddevice
4. **Speech Processing**: OpenAI Whisper for transcription
5. **AI Response**: OpenAI GPT for generating responses
6. **Voice Synthesis**: OpenAI TTS for speech output
7. **LiveKit Integration**: Optional room management and real-time communication

### Audio Pipeline

```
Microphone → Wake Word Detection → Audio Recording → Speech-to-Text → AI Processing → Text-to-Speech → Speaker
```

## Configuration Options

### Wake Words
You can customize wake words in the `setup_wake_word_detection()` method:

```python
self.porcupine = pvporcupine.create(
    access_key=self.porcupine_access_key,
    keywords=["hey computer", "hey assistant", "wake up"],  # Add your keywords
    # Or use custom wake word files:
    # keyword_paths=["./wake_words/custom_wake_word.ppn"]
)
```

### Recording Duration
Adjust the listening duration in `start_conversation()`:

```python
audio_data = await self.record_conversation_audio(duration=10)  # 10 seconds
```

### OpenAI Model Configuration
Modify the AI model settings in `get_openai_response()`:

```python
response = await self.openai_client.chat.completions.create(
    model="gpt-4",  # Use GPT-4 for better responses
    max_tokens=200,  # Longer responses
    temperature=0.5  # More deterministic responses
)
```

### Voice Configuration
Change the TTS voice in `text_to_speech_and_play()`:

```python
response = await self.openai_client.audio.speech.create(
    model="tts-1-hd",  # Higher quality TTS
    voice="nova",      # Different voice (alloy, echo, fable, onyx, nova, shimmer)
    input=text
)
```

## Troubleshooting

### Common Issues

1. **Microphone Access**: Ensure your system allows microphone access
2. **Audio Drivers**: Install proper audio drivers for your system
3. **API Keys**: Verify all API keys are correctly set in `.env`
4. **Dependencies**: Make sure all Python packages are installed

### Debug Mode
Enable debug logging by modifying the logging level:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Testing Audio
Test your microphone setup:

```python
import sounddevice as sd
print(sd.query_devices())  # List available audio devices
```

## Extending the Agent

### Adding Custom Commands
You can extend the agent to handle specific commands:

```python
async def handle_custom_command(self, text: str) -> str:
    if "time" in text.lower():
        import datetime
        return f"The current time is {datetime.datetime.now().strftime('%H:%M')}"
    return None
```

### Integration with External APIs
Add weather, news, or other API integrations:

```python
async def get_weather(self, location: str) -> str:
    # Integrate with weather API
    pass
```

### Custom Wake Words
Create custom wake words using Picovoice Console and add them to the `wake_words/` directory.

## Performance Optimization

- **Audio Quality**: Adjust sample rate and chunk size for your needs
- **Response Time**: Use faster OpenAI models (gpt-3.5-turbo vs gpt-4)
- **Memory Usage**: Implement audio buffer management for long sessions
- **Network**: Use local LiveKit server for reduced latency

## License

This project uses various third-party services and libraries. Please check their respective licenses:
- OpenAI API: [OpenAI Terms](https://openai.com/terms/)
- Porcupine: [Picovoice License](https://picovoice.ai/docs/quick-start/porcupine-python/)
- LiveKit: [Apache 2.0 License](https://github.com/livekit/livekit) 