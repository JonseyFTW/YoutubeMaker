# Python Tutorial Generator

A production-ready system for automatically generating high-quality Python tutorial videos.

## Features

- **Script Generation**: Uses OpenAI's models to generate tutorial scripts
- **Text-to-Speech Narration**: Multiple TTS options including ElevenLabs, Google TTS, and pyttsx3
- **Slide and Visualization Creation**: Creates professional slides for your tutorials
- **Code Animations**: Generates animated code demonstrations
- **Final Video Assembly**: Combines all elements into a polished final video
- **Monetization Optimization**: Adds sponsorship segments and optimal ad break points

## Installation

### Using Docker (Recommended)

1. Clone this repository
2. Configure your API keys in `config.json`
3. Build and run the Docker container:

````bash
docker-compose up --build
````

### Manual Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your API keys in `config.json`

## Configuration

The system is configured through `config.json`. Here's an example configuration:

```json
{
  "openai": {
    "api_key": "your-openai-api-key",
    "model": "o1-mini"
  },
  "tts": {
    "provider": "gtts",
    "providers": {
      "elevenlabs": {
        "api_key": "your-elevenlabs-api-key",
        "voice_id": "your-voice-id"
      },
      "gtts": {
        "language": "en",
        "slow": false
      },
      "pyttsx3": {
        "voice": "default",
        "rate": 150,
        "volume": 1.0
      }
    }
  },
  "slides": {
    "provider": "internal",
    "template": "modern-blue",
    "font_size": {
      "title": 48,
      "content": 36
    }
  },
  "code_animations": {
    "provider": "internal",
    "theme": "monokai",
    "font_size": 30,
    "animation_speed": 1.5
  },
  "video": {
    "resolution": {
      "width": 1920,
      "height": 1080
    },
    "fps": 24,
    "codec": "libx264"
  }
}
```

## Text-to-Speech Options

The system supports multiple TTS providers:

1. **ElevenLabs** (Premium quality, requires API key)
   - Provides high-quality, natural-sounding voices
   - Requires credits from ElevenLabs

2. **Google Text-to-Speech** (Free)
   - Good quality voices
   - Internet connection required
   - No API key needed

3. **pyttsx3** (Free, offline)
   - Works completely offline
   - Uses system voices
   - Lower quality than online options

### Listing Available Voices

For pyttsx3, you can list available voices with:

```bash
python list_voices.py
```

### Testing TTS Providers

You can test the different TTS providers with:

```bash
python test_tts.py --provider all
python test_tts.py --provider elevenlabs
python test_tts.py --provider gtts
python test_tts.py --provider pyttsx3
```

## Usage

Generate a tutorial with:

```bash
python tutorial_generator.py "Python List Comprehensions" --difficulty intermediate
```

Specify a TTS provider:

```bash
python tutorial_generator.py "Python List Comprehensions" --tts-provider gtts
```

## License

MIT
