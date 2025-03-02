#!/usr/bin/env python
"""
Utility script to list available TTS voices for pyttsx3.
"""

import pyttsx3
import argparse

def list_pyttsx3_voices():
    """List all available voices for pyttsx3."""
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    print(f"Found {len(voices)} voices for pyttsx3:")
    print("-" * 50)
    
    for i, voice in enumerate(voices):
        print(f"Voice #{i+1}")
        print(f"  ID: {voice.id}")
        print(f"  Name: {voice.name}")
        print(f"  Languages: {voice.languages}")
        print(f"  Gender: {voice.gender}")
        print(f"  Age: {voice.age}")
        print("-" * 50)
    
    print("\nTo use a specific voice, add its ID to the config.json file under tts.providers.pyttsx3.voice")

def main():
    parser = argparse.ArgumentParser(description="List available TTS voices")
    parser.add_argument("--provider", choices=["pyttsx3"], default="pyttsx3", 
                       help="TTS provider to list voices for")
    
    args = parser.parse_args()
    
    if args.provider == "pyttsx3":
        list_pyttsx3_voices()
    else:
        print(f"Listing voices for {args.provider} is not supported yet.")

if __name__ == "__main__":
    main()
