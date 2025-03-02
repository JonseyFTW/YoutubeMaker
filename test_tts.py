#!/usr/bin/env python
"""
Utility script to test different TTS providers.
"""

import os
import json
import argparse
from gtts import gTTS
import pyttsx3
from pydub import AudioSegment
import requests

def test_elevenlabs(text, config_path="config.json"):
    """Test ElevenLabs TTS."""
    print("Testing ElevenLabs TTS...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    elevenlabs_config = config["tts"]["providers"]["elevenlabs"]
    api_key = elevenlabs_config["api_key"]
    voice_id = elevenlabs_config["voice_id"]
    
    # Direct API call to ElevenLabs
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    print("Sending request to ElevenLabs API...")
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        # Save the audio content to a file
        output_path = "elevenlabs_test.mp3"
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        print(f"ElevenLabs TTS test successful! Audio saved to {output_path}")
        return output_path
    else:
        print(f"ElevenLabs API error: {response.status_code} - {response.text}")
        return None

def test_gtts(text):
    """Test Google Text-to-Speech."""
    print("Testing Google Text-to-Speech...")
    
    output_path = "gtts_test.mp3"
    
    # Generate speech using gTTS
    tts = gTTS(text=text, lang="en", slow=False)
    
    # Save to file
    tts.save(output_path)
    
    print(f"Google TTS test successful! Audio saved to {output_path}")
    return output_path

def test_pyttsx3(text):
    """Test pyttsx3 (offline TTS engine)."""
    print("Testing pyttsx3 TTS...")
    
    # Initialize the TTS engine
    engine = pyttsx3.init()
    
    # Set properties
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    
    # Create a temporary WAV file (pyttsx3 only supports WAV)
    temp_wav = "pyttsx3_test.wav"
    output_path = "pyttsx3_test.mp3"
    
    # Generate speech
    engine.save_to_file(text, temp_wav)
    engine.runAndWait()
    
    # Convert WAV to MP3 using pydub
    if os.path.exists(temp_wav):
        audio = AudioSegment.from_wav(temp_wav)
        audio.export(output_path, format="mp3")
        
        # Clean up temp file
        os.remove(temp_wav)
    
    print(f"pyttsx3 TTS test successful! Audio saved to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Test TTS providers")
    parser.add_argument("--provider", choices=["elevenlabs", "gtts", "pyttsx3", "all"], 
                       default="all", help="TTS provider to test")
    parser.add_argument("--text", default="This is a test of the text to speech system. How does it sound?", 
                       help="Text to convert to speech")
    parser.add_argument("--config", default="config.json", 
                       help="Path to config file (for ElevenLabs)")
    
    args = parser.parse_args()
    
    if args.provider == "elevenlabs" or args.provider == "all":
        test_elevenlabs(args.text, args.config)
        
    if args.provider == "gtts" or args.provider == "all":
        test_gtts(args.text)
        
    if args.provider == "pyttsx3" or args.provider == "all":
        test_pyttsx3(args.text)

if __name__ == "__main__":
    main()
