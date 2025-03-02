"""
PythonTutorialGenerator: A production-ready system for automatically generating
high-quality Python tutorial videos.

This system automates the entire workflow:
1. Script generation using LLMs
2. Text-to-speech narration
3. Slide and visualization creation
4. Code animations
5. Final video assembly
6. Monetization optimization
"""

import os
import sys
import json
import time
import logging
import random
import string
import tempfile
import requests
import numpy as np
import argparse
import re
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import openai
from gtts import gTTS
import pyttsx3
from pydub import AudioSegment
from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip, TextClip, CompositeVideoClip, 
    concatenate_videoclips, CompositeAudioClip, AudioClip, ColorClip
)
from moviepy.video import fx as vfx
from moviepy.config import change_settings
from pydub import AudioSegment
import uuid
import math
import elevenlabs

# Configure MoviePy to use ImageMagick
change_settings({"IMAGEMAGICK_BINARY": os.environ.get("IMAGEMAGICK_BINARY", "convert")})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tutorial_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

class TutorialGenerator:
    """
    Main orchestration class for generating Python tutorial videos.
    Coordinates all steps in the pipeline.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the tutorial generator with configuration.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config
        self.setup_directories()
        self.setup_apis()
        
    def setup_directories(self):
        """Set up necessary directories for assets and output."""
        directories = [
            "scripts", "audio", "slides", "code_animations", 
            "temp", "output"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def setup_apis(self):
        """Set up API clients for various services."""
        # OpenAI for script generation
        openai.api_key = self.config["openai"]["api_key"]
        
        # Other API setups would go here
        # self.other_api = OtherAPI(self.config["other_api"]["key"])
    
    def generate_tutorial(self, topic: str, difficulty: str = "intermediate") -> str:
        """
        Generate a complete Python tutorial video for the given topic.
        
        Args:
            topic: The Python topic to create a tutorial for
            difficulty: Target audience skill level (beginner, intermediate, advanced)
            
        Returns:
            Path to the final rendered video
        """
        try:
            logger.info(f"Starting tutorial generation for topic: {topic}")
            
            # Step 1: Generate script
            script_data = self.generate_script(topic, difficulty)
            
            # Step 2: Create TTS narration
            audio_files = self.generate_narration(script_data)
            
            # Step 3: Generate slides for theoretical sections
            slide_files = self.generate_slides(script_data)
            
            # Step 4: Create code animations
            code_animations = self.generate_code_animations(script_data)
            
            # Step 5: Assemble final video
            output_path = self.assemble_video(
                script_data, audio_files, slide_files, code_animations
            )
            
            # Step 6: Add monetization elements
            final_path = self.add_monetization_elements(output_path, script_data)
            
            logger.info(f"Tutorial generation completed successfully: {final_path}")
            return final_path
            
        except Exception as e:
            logger.exception(f"Tutorial generation failed: {e}")
            raise
    
    def generate_script(self, topic: str, difficulty: str) -> Dict:
        """
        Generate a structured script for the tutorial using LLM.
        
        Args:
            topic: Python topic to explain
            difficulty: Target audience skill level
            
        Returns:
            Structured script data with sections, code examples, etc.
        """
        logger.info(f"Generating script for {topic} ({difficulty})")
        
        # Construct the prompt template
        prompt = f"""
        Create a detailed Python tutorial script for "{topic}" at {difficulty} level.
        
        The tutorial should:
        1. Have a clear introduction explaining what {topic} is and why it's useful
        2. Break down the explanation into logical sections
        3. Include practical code examples that demonstrate each concept
        4. Add real-world context to make it engaging
        5. Have a conclusion summarizing key points
        
        Format the response as JSON with these sections:
        - title: The tutorial title
        - introduction: Opening explanation
        - sections: Array of objects with:
          - title: Section title
          - content: Explanatory text
          - code_example: Python code demonstrating the section (if applicable)
          - code_explanation: Explanation of what the code does
        - conclusion: Summary and takeaways
        - total_runtime_estimate: Estimated runtime in minutes (aim for 10-15 minutes)
        
        For code examples, ensure they're complete, correct, and follow PEP 8.
        """
        
        # Call OpenAI API
        try:
            response = openai.ChatCompletion.create(
                model=self.config["openai"]["model"],
                messages=[
                    {"role": "user", "content": "You are an expert Python educator who creates clear, engaging tutorials. Please provide a tutorial script in JSON format with the following structure: {\"title\": \"Python Decorators\", \"introduction\": \"...\", \"sections\": [{\"title\": \"Section Title\", \"content\": \"Explanatory text\", \"code_example\": \"Python code\", \"code_explanation\": \"Narration that explains the code conceptually WITHOUT reading every character\"}], \"conclusion\": \"...\", \"total_runtime_estimate\": 15}. IMPORTANT: For code_explanation, DO NOT read the code verbatim. Instead, explain the concepts and what the code is doing at a higher level. " + prompt}
                ],
            )
            
            # Extract and parse the script
            script_json = response.choices[0].message.content
            
            # Try to find JSON content in the response
            try:
                # Look for JSON content between triple backticks if present
                import re
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', script_json)
                if json_match:
                    script_json = json_match.group(1).strip()
                
                script_data = json.loads(script_json)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a basic structure from the text
                logger.warning("Failed to parse JSON response, creating basic structure")
                script_data = {
                    "title": f"Python {topic}",
                    "introduction": "Introduction to " + topic,
                    "sections": [
                        {
                            "title": "Overview",
                            "content": script_json[:1000] if len(script_json) > 1000 else script_json
                        }
                    ],
                    "conclusion": "Thank you for watching this tutorial.",
                    "code_examples": [],
                    "total_runtime_estimate": 10
                }
            
            # Save script to file
            script_path = os.path.join("scripts", f"{topic.replace(' ', '_')}.json")
            with open(script_path, "w") as f:
                json.dump(script_data, f, indent=2)
            
            logger.info(f"Script generated and saved to {script_path}")
            return script_data
            
        except Exception as e:
            logger.error(f"Script generation failed: {e}")
            raise
    
    def generate_narration(self, script_data: Dict) -> Dict[str, str]:
        """
        Generate TTS narration for all sections of the script.
        
        Args:
            script_data: The structured script data
            
        Returns:
            Dictionary mapping section IDs to audio file paths
        """
        logger = logging.getLogger(__name__)
        
        logger.info("Generating TTS narration")
        
        # Extract narration texts from script data
        narration_texts = self._extract_narration_texts(script_data)
        
        # Dictionary to store audio file paths
        audio_files = {}
        
        # Generate TTS for each section
        for section_id, text in narration_texts.items():
            try:
                # Create a unique filename for this section
                audio_path = os.path.join("audio", f"{section_id}.mp3")
                
                # If audio file already exists, use it
                if os.path.exists(audio_path):
                    audio_files[section_id] = audio_path
                    logger.info(f"Using existing narration for {section_id}")
                    continue
                
                # Generate TTS using the selected provider
                audio_files[section_id] = self._generate_tts(section_id, text, audio_path)
                logger.info(f"Generated narration for {section_id}")
                
            except Exception as e:
                logger.error(f"TTS generation failed for section {section_id}: {str(e)}")
                # Create a minimal silent audio file as a last resort
                try:
                    audio_path = os.path.join("audio", f"{section_id}.mp3")
                    minimal_audio = AudioSegment.silent(duration=1000)
                    minimal_audio.export(audio_path, format="mp3")
                    audio_files[section_id] = audio_path
                except Exception as inner_e:
                    logger.error(f"Failed to create minimal audio for {section_id}: {str(inner_e)}")
        
        return audio_files
    
    def _extract_narration_texts(self, script_data: Dict) -> Dict[str, str]:
        """
        Extract text for each section that needs narration.
        
        Args:
            script_data: The structured script data
            
        Returns:
            Dictionary mapping section IDs to narration text
        """
        narration_texts = {
            "intro": script_data["introduction"]
        }
        
        for i, section in enumerate(script_data["sections"]):
            section_id = f"section_{i}"
            narration_texts[section_id] = section["content"]
            
            # Check if this section has a code example and explanation
            if "code_example" in section and section["code_example"]:
                code_id = f"code_{i}"
                
                # Use the code_explanation if available, otherwise create a generic explanation
                if "code_explanation" in section and section["code_explanation"]:
                    narration_texts[code_id] = section["code_explanation"]
                else:
                    # Create a generic explanation if none is provided
                    narration_texts[code_id] = f"Let's look at this code example for {section['title']}."
        
        narration_texts["conclusion"] = script_data["conclusion"]
        
        return narration_texts
    
    def _generate_tts(self, section_id: str, text: str, audio_path: str) -> str:
        """Generate TTS narration for a section"""
        logger = logging.getLogger(__name__)
        
        # Get TTS provider from config
        tts_provider = self.config["tts"]["provider"]
        
        logger.info(f"Using TTS provider: {tts_provider}")
        
        # Call the appropriate TTS generation method based on the provider
        if tts_provider == "elevenlabs":
            return self._generate_elevenlabs_tts(section_id, text, audio_path)
        elif tts_provider == "gtts":
            return self._generate_gtts_tts(section_id, text, audio_path)
        elif tts_provider == "pyttsx3":
            return self._generate_pyttsx3_tts(section_id, text, audio_path)
        elif tts_provider == "silent":
            return self._generate_silent_tts(section_id, text, audio_path)
        else:
            logger.warning(f"Unknown TTS provider: {tts_provider}, falling back to silent")
            return self._generate_silent_tts(section_id, text, audio_path)
    
    def _generate_elevenlabs_tts(self, section_id: str, text: str, audio_path: str) -> str:
        """Generate TTS using ElevenLabs API"""
        logger = logging.getLogger(__name__)
        
        try:
            # Get ElevenLabs configuration
            elevenlabs_config = self.config["tts"]["providers"]["elevenlabs"]
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
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                # Save the audio content to a file
                with open(audio_path, "wb") as f:
                    f.write(response.content)
                
                logger.info(f"Generated ElevenLabs narration for {section_id}")
                return audio_path
            else:
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                # Fall back to silent audio if API fails
                logger.warning("Falling back to silent audio generation")
                return self._generate_silent_tts(section_id, text, audio_path)
                
        except Exception as e:
            logger.error(f"ElevenLabs API error: {str(e)}")
            # Fall back to silent audio if API fails
            logger.warning("Falling back to silent audio generation")
            return self._generate_silent_tts(section_id, text, audio_path)
    
    def _generate_gtts_tts(self, section_id: str, text: str, audio_path: str) -> str:
        """Generate TTS using Google Text-to-Speech"""
        logger = logging.getLogger(__name__)
        
        try:
            # Get gTTS configuration
            gtts_config = self.config["tts"]["providers"]["gtts"]
            language = gtts_config.get("language", "en")
            slow = gtts_config.get("slow", False)
            
            # Generate speech using gTTS
            tts = gTTS(text=text, lang=language, slow=slow)
            
            # Save to temporary file first (to avoid issues with partial writes)
            temp_file = f"{audio_path}.temp"
            tts.save(temp_file)
            
            # Convert to proper format with pydub to ensure compatibility
            audio = AudioSegment.from_file(temp_file)
            audio.export(audio_path, format="mp3")
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            logger.info(f"Generated gTTS narration for {section_id}")
            return audio_path
            
        except Exception as e:
            logger.error(f"gTTS error: {str(e)}")
            # Fall back to silent audio if gTTS fails
            logger.warning("Falling back to silent audio generation")
            return self._generate_silent_tts(section_id, text, audio_path)
    
    def _generate_pyttsx3_tts(self, section_id: str, text: str, audio_path: str) -> str:
        """Generate TTS using pyttsx3 (offline TTS engine)"""
        logger = logging.getLogger(__name__)
        
        try:
            # Get pyttsx3 configuration
            pyttsx3_config = self.config["tts"]["providers"]["pyttsx3"]
            voice = pyttsx3_config.get("voice", "default")
            rate = pyttsx3_config.get("rate", 150)
            volume = pyttsx3_config.get("volume", 1.0)
            
            # Initialize the TTS engine
            engine = pyttsx3.init()
            
            # Set properties
            engine.setProperty('rate', rate)
            engine.setProperty('volume', volume)
            
            # Set voice if not default
            if voice != "default":
                voices = engine.getProperty('voices')
                for v in voices:
                    if voice in v.id:
                        engine.setProperty('voice', v.id)
                        break
            
            # Create a temporary WAV file (pyttsx3 only supports WAV)
            temp_wav = f"{audio_path}.wav"
            
            # Generate speech
            engine.save_to_file(text, temp_wav)
            engine.runAndWait()
            
            # Convert WAV to MP3 using pydub
            if os.path.exists(temp_wav):
                audio = AudioSegment.from_wav(temp_wav)
                audio.export(audio_path, format="mp3")
                
                # Clean up temp file
                os.remove(temp_wav)
                
            logger.info(f"Generated pyttsx3 narration for {section_id}")
            return audio_path
            
        except Exception as e:
            logger.error(f"pyttsx3 error: {str(e)}")
            # Fall back to silent audio if pyttsx3 fails
            logger.warning("Falling back to silent audio generation")
            return self._generate_silent_tts(section_id, text, audio_path)
    
    def _generate_silent_tts(self, section_id: str, text: str, audio_path: str, minimal: bool = False) -> str:
        """Generate silent audio as a fallback"""
        logger = logging.getLogger(__name__)
        
        try:
            # Calculate duration based on word count
            word_count = len(text.split())
            
            if minimal:
                # Just create a very short silent clip as a last resort
                duration_ms = 1000
            else:
                # Create a more realistic duration based on reading speed
                duration_ms = max(1000, word_count * 200)  # At least 1 second, ~200ms per word
            
            # Create silent audio
            silent_audio = AudioSegment.silent(duration=duration_ms)
            
            # Export the audio segment to the specified path
            silent_audio.export(audio_path, format="mp3")
            
            logger.info(f"Generated silent narration for {section_id}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Silent audio generation failed: {str(e)}")
            
            # Create an absolute minimal audio file as a last resort
            try:
                minimal_audio = AudioSegment.silent(duration=1000)
                minimal_audio.export(audio_path, format="mp3")
                
                return audio_path
            except Exception as e:
                logger.error(f"Minimal audio generation failed: {str(e)}")
                raise
    
    def generate_slides(self, script_data: Dict) -> Dict[str, str]:
        """
        Generate slides for theoretical explanations.
        
        Args:
            script_data: The structured script data
            
        Returns:
            Dictionary mapping section IDs to slide image files
        """
        logger.info("Generating slides for theoretical sections")
        
        slide_files = {}
        
        # Generate title slide
        title_slide = self._generate_title_slide(script_data["title"])
        slide_files["title"] = title_slide
        
        # Generate intro slide
        intro_slide = self._generate_content_slide(
            "Introduction", script_data["introduction"], 0
        )
        slide_files["intro"] = intro_slide
        
        # Generate section slides
        for i, section in enumerate(script_data["sections"]):
            section_id = f"section_{i}"
            section_slide = self._generate_content_slide(
                section["title"], section["content"], i+1
            )
            slide_files[section_id] = section_slide
        
        # Generate conclusion slide
        conclusion_slide = self._generate_content_slide(
            "Conclusion", script_data["conclusion"], len(script_data["sections"])+1
        )
        slide_files["conclusion"] = conclusion_slide
        
        return slide_files
    
    def _generate_title_slide(self, title: str) -> str:
        """
        Generate a title slide for the tutorial.
        
        Args:
            title: The title of the tutorial
            
        Returns:
            Path to the generated slide image
        """
        # Create a unique filename for this slide
        slide_id = str(uuid.uuid4())[:8]
        slide_path = os.path.join("slides", f"title_{slide_id}.png")
        
        # Create slide using PIL
        width, height = 1280, 720
        
        # Create a gradient background with a more dynamic color scheme
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw a more interesting gradient background (dark blue to purple)
        for y in range(height):
            # Calculate gradient color
            r = int(20 + (y / height) * 40)  # Dark to slightly less dark red
            g = int(10 + (y / height) * 20)  # Very dark to dark green
            b = int(80 + (y / height) * 100)  # Medium blue to purple
            
            # Draw horizontal line with calculated color
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        # Add decorative elements
        
        # Add a subtle pattern of dots in the background
        for i in range(50):  # Add 50 random dots
            dot_x = np.random.randint(0, width)
            dot_y = np.random.randint(0, height)
            dot_size = np.random.randint(2, 6)
            dot_color = (
                min(255, r + np.random.randint(50, 150)),
                min(255, g + np.random.randint(50, 150)),
                min(255, b + np.random.randint(50, 150)),
                np.random.randint(100, 200)  # Alpha transparency
            )
            
            # Draw the dot
            draw.ellipse(
                [(dot_x - dot_size, dot_y - dot_size), 
                 (dot_x + dot_size, dot_y + dot_size)], 
                fill=dot_color
            )
        
        # Add a decorative border with rounded corners
        border_width = 15
        corner_radius = 30
        
        # Draw the border with rounded corners
        # Top and bottom horizontal lines
        draw.line([(corner_radius, border_width), (width - corner_radius, border_width)], 
                 fill=(255, 255, 255), width=3)
        draw.line([(corner_radius, height - border_width), (width - corner_radius, height - border_width)], 
                 fill=(255, 255, 255), width=3)
        
        # Left and right vertical lines
        draw.line([(border_width, corner_radius), (border_width, height - corner_radius)], 
                 fill=(255, 255, 255), width=3)
        draw.line([(width - border_width, corner_radius), (width - border_width, height - corner_radius)], 
                 fill=(255, 255, 255), width=3)
        
        # Draw the rounded corners
        draw.arc([(border_width - corner_radius, border_width - corner_radius), 
                  (border_width + corner_radius, border_width + corner_radius)], 
                 180, 270, fill=(255, 255, 255), width=3)
        
        draw.arc([(width - border_width - corner_radius, border_width - corner_radius), 
                  (width - border_width + corner_radius, border_width + corner_radius)], 
                 270, 0, fill=(255, 255, 255), width=3)
        
        draw.arc([(border_width - corner_radius, height - border_width - corner_radius), 
                  (border_width + corner_radius, height - border_width + corner_radius)], 
                 90, 180, fill=(255, 255, 255), width=3)
        
        draw.arc([(width - border_width - corner_radius, height - border_width - corner_radius), 
                  (width - border_width + corner_radius, height - border_width + corner_radius)], 
                 0, 90, fill=(255, 255, 255), width=3)
        
        # Try to load a nicer font with better fallback mechanism
        title_font = None
        subtitle_font = None
        
        # List of fonts to try in order of preference
        font_options = [
            os.path.join("assets", "fonts", "Arial.ttf"),
            os.path.join("assets", "fonts", "Roboto-Bold.ttf"),
            "arial.ttf",
            "calibri.ttf",
            "verdana.ttf"
        ]
        
        # Try each font until one works
        for font_path in font_options:
            try:
                if os.path.exists(font_path):
                    title_font = ImageFont.truetype(font_path, 72)
                    subtitle_font = ImageFont.truetype(font_path, 36)
                    break
            except Exception:
                continue
        
        # If no font worked, use default
        if title_font is None:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
        
        # Wrap title text if it's too long
        max_title_width = width - 200  # 100px margin on each side
        wrapped_title = []
        current_line = ""
        
        # Split the title into words
        words = title.split()
        
        for word in words:
            # Check if adding this word would make the line too long
            test_line = current_line + " " + word if current_line else word
            line_width = draw.textlength(test_line, font=title_font)
            
            if line_width <= max_title_width:
                current_line = test_line
            else:
                # Line is full, add it to wrapped lines and start a new line
                if current_line:
                    wrapped_title.append(current_line)
                current_line = word
        
        # Add the last line
        if current_line:
            wrapped_title.append(current_line)
        
        # If title is still empty (e.g., only whitespace), use the original
        if not wrapped_title:
            wrapped_title = [title]
        
        # Calculate vertical position for the title
        title_height = len(wrapped_title) * 80  # Approximate height of all lines
        title_y_start = (height - title_height) // 2 - 40  # Center vertically with some offset
        
        # Draw each line of the title
        for i, line in enumerate(wrapped_title):
            # Calculate text width to center this line
            line_width = draw.textlength(line, font=title_font)
            line_x = (width - line_width) // 2
            line_y = title_y_start + i * 80
            
            # Draw text with enhanced shadow/glow effect
            shadow_offset = 3
            shadow_color = (100, 50, 150)  # Purple shadow
            
            # Draw multiple shadow layers for a glow effect
            for offset in range(1, 4):
                draw.text((line_x + offset, line_y + offset), line, 
                         fill=(*shadow_color, 100), font=title_font)
                draw.text((line_x - offset, line_y + offset), line, 
                         fill=(*shadow_color, 100), font=title_font)
                draw.text((line_x + offset, line_y - offset), line, 
                         fill=(*shadow_color, 100), font=title_font)
                draw.text((line_x - offset, line_y - offset), line, 
                         fill=(*shadow_color, 100), font=title_font)
            
            # Draw main text
            draw.text((line_x, line_y), line, 
                     fill=(255, 255, 255), font=title_font)
        
        # Add a subtitle
        subtitle = "Python Tutorial"
        subtitle_width = draw.textlength(subtitle, font=subtitle_font)
        subtitle_x = (width - subtitle_width) // 2
        subtitle_y = title_y_start + title_height + 20
        
        # Draw subtitle with shadow
        shadow_offset = 2
        # Draw shadow first
        draw.text((subtitle_x + shadow_offset, subtitle_y + shadow_offset), subtitle, 
                 fill=(0, 0, 50), font=subtitle_font)
        # Draw main subtitle
        draw.text((subtitle_x, subtitle_y), subtitle, 
                 fill=(200, 200, 255), font=subtitle_font)
        
        # Add decorative elements at the bottom
        icon_size = 40
        icon_y = height - 100
        icon_spacing = 120
        
        # Calculate starting position to center the icons
        num_icons = 5
        total_width = (num_icons - 1) * icon_spacing
        icon_start_x = (width - total_width) // 2
        
        # Draw 5 programming-related icons
        icons = [
            {"name": "Python", "color": (53, 114, 165)},
            {"name": "Code", "color": (120, 170, 210)},
            {"name": "Learn", "color": (180, 180, 220)},
            {"name": "Build", "color": (150, 120, 200)},
            {"name": "Create", "color": (100, 80, 180)}
        ]
        
        for i, icon in enumerate(icons):
            icon_x = icon_start_x + i * icon_spacing
            
            # Draw icon background
            draw.ellipse(
                [(icon_x - icon_size, icon_y - icon_size), 
                 (icon_x + icon_size, icon_y + icon_size)], 
                fill=icon["color"]
            )
            
            # Draw icon text
            icon_font = subtitle_font
            icon_text = icon["name"][0]  # First letter of the name
            text_width = draw.textlength(icon_text, font=icon_font)
            text_x = icon_x - text_width // 2
            
            draw.text((text_x, icon_y - 15), icon_text, 
                     fill=(255, 255, 255), font=icon_font)
        
        # Save the image
        os.makedirs(os.path.dirname(slide_path), exist_ok=True)
        img.save(slide_path)
        
        return slide_path
        
    def _generate_content_slide(self, heading: str, content: str, index: int) -> str:
        """
        Generate a content slide for the tutorial.
        
        Args:
            heading: The heading of the slide
            content: The content of the slide
            index: The index of the slide
            
        Returns:
            Path to the generated slide image
        """
        # Create a unique filename for this slide
        slide_id = str(uuid.uuid4())[:8]
        slide_path = os.path.join("slides", f"content_{slide_id}.png")
        
        # Create slide using PIL
        width, height = 1280, 720
        
        # Create a gradient background with colors based on slide index
        # This creates a different color scheme for each slide to add variety
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Generate a color scheme based on the index
        # This creates a subtle variation between slides while maintaining a cohesive look
        base_hue = (index * 30) % 360  # Rotate through hues
        
        # Convert HSV to RGB for the gradient
        def hsv_to_rgb(h, s, v):
            h = h / 360
            c = v * s
            x = c * (1 - abs((h * 6) % 2 - 1))
            m = v - c
            
            if 0 <= h < 1/6:
                r, g, b = c, x, 0
            elif 1/6 <= h < 2/6:
                r, g, b = x, c, 0
            elif 2/6 <= h < 3/6:
                r, g, b = 0, c, x
            elif 3/6 <= h < 4/6:
                r, g, b = 0, x, c
            elif 4/6 <= h < 5/6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
        
        # Draw gradient background
        for y in range(height):
            # Calculate gradient colors with a subtle shift based on position
            progress = y / height
            
            # Adjust saturation and value based on position
            saturation = 0.7 - (progress * 0.3)  # Decrease saturation slightly toward bottom
            value = 0.3 + (progress * 0.1)  # Slightly increase brightness toward bottom
            
            # Get RGB color
            color = hsv_to_rgb(base_hue, saturation, value)
            
            # Draw horizontal line with calculated color
            draw.line([(0, y), (width, y)], fill=color)
        
        # Add a decorative border
        border_width = 10
        
        # Use a color that complements the background
        border_color = hsv_to_rgb((base_hue + 180) % 360, 0.6, 0.8)  # Complementary color
        
        draw.rectangle(
            [(border_width, border_width), (width - border_width, height - border_width)],
            outline=border_color,
            width=2
        )
        
        # Try to load fonts with better fallback mechanism
        heading_font = None
        content_font = None
        
        # List of fonts to try in order of preference
        font_options = [
            os.path.join("assets", "fonts", "Arial.ttf"),
            os.path.join("assets", "fonts", "Roboto-Bold.ttf"),
            "arial.ttf",
            "calibri.ttf",
            "verdana.ttf"
        ]
        
        # Try each font until one works
        for font_path in font_options:
            try:
                if os.path.exists(font_path):
                    heading_font = ImageFont.truetype(font_path, 48)
                    content_font = ImageFont.truetype(font_path, 32)
                    break
            except Exception:
                continue
        
        # If no font worked, use default
        if heading_font is None:
            heading_font = ImageFont.load_default()
            content_font = ImageFont.load_default()
        
        # Calculate text width to center heading
        heading_width = draw.textlength(heading, font=heading_font)
        heading_x = (width - heading_width) // 2
        
        # Draw heading with shadow effect
        shadow_offset = 2
        # Draw shadow first
        draw.text((heading_x + shadow_offset, 80 + shadow_offset), heading, 
                 fill=(0, 0, 50), font=heading_font)
        # Draw main heading
        draw.text((heading_x, 80), heading, 
                 fill=(255, 255, 255), font=heading_font)
        
        # Add a decorative line under the heading
        line_y = 140
        line_margin = 100
        draw.line([(line_margin, line_y), (width - line_margin, line_y)], 
                 fill=border_color, width=2)
        
        # Add slide number in a decorative circle
        circle_radius = 30
        circle_x = width - 60
        circle_y = 60
        
        # Draw circle background
        draw.ellipse(
            [(circle_x - circle_radius, circle_y - circle_radius), 
             (circle_x + circle_radius, circle_y + circle_radius)], 
            fill=border_color
        )
        
        # Draw slide number
        slide_num = str(index + 1)
        num_width = draw.textlength(slide_num, font=content_font)
        num_x = circle_x - num_width // 2
        
        draw.text((num_x, circle_y - 15), slide_num, 
                 fill=(255, 255, 255), font=content_font)
        
        # Process content text - detect bullet points and format accordingly
        content_lines = []
        
        # Split content into lines and process each line
        raw_lines = content.strip().split("\n")
        
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with bullet point indicators
            is_bullet = False
            indent = 0
            
            # Check for common bullet point formats
            bullet_patterns = [
                (r'^\s*•\s+(.+)$', '•'),
                (r'^\s*\*\s+(.+)$', '•'),
                (r'^\s*-\s+(.+)$', '•'),
                (r'^\s*\d+\.\s+(.+)$', None)  # Numbered list
            ]
            
            for pattern, replacement in bullet_patterns:
                match = re.match(pattern, line)
                if match:
                    is_bullet = True
                    if replacement:
                        # For bullet points, use the replacement character
                        line_text = match.group(1)
                        content_lines.append({
                            'text': line_text,
                            'is_bullet': True,
                            'bullet_char': replacement,
                            'indent': indent
                        })
                    else:
                        # This is a numbered list item, no need to add anything
                        content_lines.append({
                            'text': line,
                            'is_bullet': True,
                            'bullet_char': None,
                            'indent': indent
                        })
                    break
            
            # If not a bullet point, add as regular text
            if not is_bullet:
                content_lines.append({
                    'text': line,
                    'is_bullet': False,
                    'bullet_char': None,
                    'indent': 0
                })
        
        # Wrap text to fit the slide width
        max_text_width = width - 200  # 100px margin on each side
        wrapped_lines = []
        
        for line_info in content_lines:
            line = line_info['text']
            is_bullet = line_info['is_bullet']
            bullet_char = line_info['bullet_char']
            indent = line_info['indent']
            
            # Calculate available width (accounting for bullet point indentation)
            available_width = max_text_width - (indent * 20)
            if is_bullet:
                available_width -= 40  # Space for bullet character
            
            # Split the line into words
            words = line.split()
            current_line = ""
            
            for word in words:
                # Check if adding this word would make the line too long
                test_line = current_line + " " + word if current_line else word
                line_width = draw.textlength(test_line, font=content_font)
                
                if line_width <= available_width:
                    current_line = test_line
                else:
                    # Line is full, add it to wrapped lines and start a new line
                    if current_line:
                        wrapped_lines.append({
                            'text': current_line,
                            'is_bullet': is_bullet and len(wrapped_lines) == 0,  # Only first line gets bullet
                            'bullet_char': bullet_char if len(wrapped_lines) == 0 else None,
                            'indent': indent + (1 if is_bullet and len(wrapped_lines) > 0 else 0)
                        })
                    current_line = word
            
            # Add the last line
            if current_line:
                wrapped_lines.append({
                    'text': current_line,
                    'is_bullet': is_bullet and len(wrapped_lines) == 0,
                    'bullet_char': bullet_char if len(wrapped_lines) == 0 else None,
                    'indent': indent + (1 if is_bullet and len(wrapped_lines) > 0 else 0)
                })
        
        # Draw content text
        y_position = 180
        line_height = 40
        
        for line_info in wrapped_lines:
            line_text = line_info['text']
            is_bullet = line_info['is_bullet']
            bullet_char = line_info['bullet_char']
            indent = line_info['indent']
            
            # Calculate x position based on indentation
            line_x = 100 + (indent * 20)
            
            # Draw bullet point if needed
            if is_bullet:
                if bullet_char:
                    # Draw custom bullet character
                    draw.text((line_x - 30, y_position), bullet_char, 
                             fill=(255, 255, 255), font=content_font)
                else:
                    # This is a numbered list item, no need to add anything
                    pass
            
            # Draw line text with shadow for better readability
            draw.text((line_x + shadow_offset, y_position + shadow_offset), line_text, 
                     fill=(0, 0, 50), font=content_font)
            draw.text((line_x, y_position), line_text, 
                     fill=(255, 255, 255), font=content_font)
            
            y_position += line_height
        
        # Add a decorative element in the bottom corner
        corner_size = 30
        draw.ellipse(
            [(width-80, height-80), (width-20, height-20)], 
            fill=border_color, outline=(255, 255, 255)
        )
        
        # Save the image
        os.makedirs(os.path.dirname(slide_path), exist_ok=True)
        img.save(slide_path)
        
        return slide_path
        
    def _generate_sponsorship_slide(self, sponsor_name: str) -> str:
        """
        Generate a sponsorship slide for the tutorial.
        
        Args:
            sponsor_name: The name of the sponsor
            
        Returns:
            Path to the generated slide image
        """
        # Create a unique filename for this slide
        slide_id = str(uuid.uuid4())[:8]
        slide_path = os.path.join("slides", f"sponsor_{slide_id}.png")
        
        # Create slide using PIL
        width, height = 1280, 720
        
        # Create a premium-looking gradient background (gold to dark blue)
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw radial gradient background for a more premium look
        center_x, center_y = width // 2, height // 2
        max_radius = int(math.sqrt(width**2 + height**2) / 2)
        
        for y in range(height):
            for x in range(width):
                # Calculate distance from center
                dx = x - center_x
                dy = y - center_y
                distance = math.sqrt(dx**2 + dy**2)
                
                # Normalize distance
                normalized_distance = distance / max_radius
                
                # Calculate color based on distance from center
                r = int(180 - normalized_distance * 140)  # Gold to darker
                g = int(150 - normalized_distance * 130)  # Gold to darker
                b = int(50 + normalized_distance * 80)    # Gold to blue
                
                # Set pixel color
                img.putpixel((x, y), (r, g, b))
        
        # Add a premium decorative border with gold color
        border_width = 20
        
        # Outer border
        draw.rectangle(
            [(border_width, border_width), (width - border_width, height - border_width)],
            outline=(255, 215, 0),  # Gold color
            width=5
        )
        
        # Inner border
        draw.rectangle(
            [(border_width + inner_margin, border_width + inner_margin), 
             (width - border_width - inner_margin, height - border_width - inner_margin)],
            outline=(180, 150, 50),  # Darker gold
            width=2
        )
        
        # Add decorative corner elements
        corner_size = 50
        
        # Top-left corner decoration
        draw.polygon(
            [(border_width, border_width), 
             (border_width + corner_size, border_width),
             (border_width, border_width + corner_size)],
            fill=(255, 215, 0)
        )
        
        # Top-right corner decoration
        draw.polygon(
            [(width - border_width, border_width), 
             (width - border_width - corner_size, border_width),
             (width - border_width, border_width + corner_size)],
            fill=(255, 215, 0)
        )
        
        # Bottom-left corner decoration
        draw.polygon(
            [(border_width, height - border_width), 
             (border_width + corner_size, height - border_width),
             (border_width, height - border_width - corner_size)],
            fill=(255, 215, 0)
        )
        
        # Bottom-right corner decoration
        draw.polygon(
            [(width - border_width, height - border_width), 
             (width - border_width - corner_size, height - border_width),
             (width - border_width, height - border_width - corner_size)],
            fill=(255, 215, 0)
        )
        
        # Try to load a premium-looking font with better fallback mechanism
        title_font = None
        sponsor_font = None
        
        # List of fonts to try in order of preference
        font_options = [
            os.path.join("assets", "fonts", "Arial.ttf"),
            os.path.join("assets", "fonts", "Roboto-Bold.ttf"),
            "arial.ttf",
            "calibri.ttf",
            "georgia.ttf",
            "verdana.ttf"
        ]
        
        # Try each font until one works
        for font_path in font_options:
            try:
                if os.path.exists(font_path):
                    title_font = ImageFont.truetype(font_path, 48)
                    sponsor_font = ImageFont.truetype(font_path, 72)
                    break
            except Exception:
                continue
        
        # If no font worked, use default
        if title_font is None:
            title_font = ImageFont.load_default()
            sponsor_font = ImageFont.load_default()
        
        # Draw "This tutorial is sponsored by" text
        sponsor_text = "This tutorial is sponsored by"
        sponsor_text_width = draw.textlength(sponsor_text, font=title_font)
        sponsor_text_x = (width - sponsor_text_width) // 2
        sponsor_text_y = 220
        
        # Draw text with enhanced shadow/glow effect
        shadow_offset = 2
        shadow_color = (100, 80, 0)  # Dark gold shadow
        
        # Draw multiple shadow layers for a glow effect
        for offset in range(1, 4):
            draw.text((sponsor_text_x + offset, sponsor_text_y + offset), sponsor_text, 
                     fill=(*shadow_color, 150), font=title_font)
            draw.text((sponsor_text_x - offset, sponsor_text_y + offset), sponsor_text, 
                     fill=(*shadow_color, 150), font=title_font)
        
        # Draw main text
        draw.text((sponsor_text_x, sponsor_text_y), sponsor_text, 
                 fill=(255, 255, 255), font=title_font)
        
        # Draw sponsor name with larger font and gold color
        # Wrap sponsor name if it's too long
        max_sponsor_width = width - 200  # 100px margin on each side
        
        # Check if sponsor name needs wrapping
        sponsor_name_width = draw.textlength(sponsor_name, font=sponsor_font)
        
        if sponsor_name_width <= max_sponsor_width:
            # No wrapping needed
            sponsor_lines = [sponsor_name]
        else:
            # Need to wrap the sponsor name
            words = sponsor_name.split()
            sponsor_lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                line_width = draw.textlength(test_line, font=sponsor_font)
                
                if line_width <= max_sponsor_width:
                    current_line = test_line
                else:
                    if current_line:
                        sponsor_lines.append(current_line)
                    current_line = word
            
            # Add the last line
            if current_line:
                sponsor_lines.append(current_line)
        
        # Calculate vertical position for sponsor name
        sponsor_height = len(sponsor_lines) * 80  # Approximate height of all lines
        sponsor_y_start = 300
        
        # Draw each line of the sponsor name
        for i, line in enumerate(sponsor_lines):
            # Calculate text width to center this line
            line_width = draw.textlength(line, font=sponsor_font)
            line_x = (width - line_width) // 2
            line_y = sponsor_y_start + i * 80
            
            # Draw sponsor name with enhanced shadow and gold color
            # Draw shadow first with multiple layers for a premium look
            for offset in range(1, 5):
                draw.text((line_x + offset, line_y + offset), line, 
                         fill=(80, 60, 0, 200), font=sponsor_font)
            
            # Draw main text with gold gradient effect
            # We'll simulate a gradient by drawing the text multiple times with slightly different colors
            gold_colors = [
                (255, 215, 0),  # Bright gold
                (255, 223, 0),  # Slightly brighter
                (255, 200, 0),  # Slightly darker
            ]
            
            for i, color in enumerate(gold_colors):
                offset = i - 1  # -1, 0, 1
                draw.text((line_x + offset, line_y + offset), line, 
                         fill=color, font=sponsor_font)
        
        # Add decorative elements - stars around the sponsor name
        star_radius = 15
        star_color = (255, 215, 0)  # Gold
        
        # Draw 5 stars in a circle around the sponsor name
        center_x, center_y = width // 2, sponsor_y_start + sponsor_height // 2
        circle_radius = 350
        
        for i in range(5):
            angle = i * (2 * 3.14159 / 5)  # Evenly space 5 stars
            star_x = center_x + int(circle_radius * math.cos(angle))
            star_y = center_y + int(circle_radius * 0.5 * math.sin(angle))
            
            # Draw a star shape
            # Define star points
            outer_radius = star_radius
            inner_radius = star_radius * 0.4
            star_points = []
            
            for j in range(10):
                # Alternate between outer and inner points
                current_radius = outer_radius if j % 2 == 0 else inner_radius
                point_angle = j * math.pi / 5
                
                x = star_x + current_radius * math.cos(point_angle)
                y = star_y + current_radius * math.sin(point_angle)
                star_points.append((x, y))
            
            # Draw the star
            draw.polygon(star_points, fill=star_color)
        
        # Add a "premium sponsor" badge
        badge_radius = 70
        badge_x = width - 120
        badge_y = 120
        
        # Draw badge circle
        draw.ellipse(
            [(badge_x - badge_radius, badge_y - badge_radius), 
             (badge_x + badge_radius, badge_y + badge_radius)], 
            fill=(180, 0, 0),  # Red
            outline=(255, 215, 0),  # Gold
            width=3
        )
        
        # Draw badge text
        badge_text = "PREMIUM"
        badge_text2 = "SPONSOR"
        
        badge_font = title_font
        
        # Center text in badge
        badge_text_width = draw.textlength(badge_text, font=badge_font)
        badge_text_x = badge_x - badge_text_width // 2
        
        badge_text2_width = draw.textlength(badge_text2, font=badge_font)
        badge_text2_x = badge_x - badge_text2_width // 2
        
        # Draw badge text with shadow
        draw.text((badge_text_x + 2, badge_y - 25 + 2), badge_text, 
                 fill=(100, 0, 0), font=badge_font)
        draw.text((badge_text_x, badge_y - 25), badge_text, 
                 fill=(255, 255, 255), font=badge_font)
        
        draw.text((badge_text2_x + 2, badge_y + 15 + 2), badge_text2, 
                 fill=(100, 0, 0), font=badge_font)
        draw.text((badge_text2_x, badge_y + 15), badge_text2, 
                 fill=(255, 255, 255), font=badge_font)
        
        # Save the image
        os.makedirs(os.path.dirname(slide_path), exist_ok=True)
        img.save(slide_path)
        
        return slide_path
    
    def generate_code_animations(self, script_data: Dict) -> Dict[str, str]:
        """
        Generate animated code demonstrations.
        
        Args:
            script_data: The structured script data
            
        Returns:
            Dictionary mapping section IDs to code animation video files
        """
        logger.info("Generating code animations")
        
        code_animations = {}
        
        for i, section in enumerate(script_data["sections"]):
            if "code_example" in section and section["code_example"]:
                try:
                    code_id = f"code_{i}"
                    code = section["code_example"]
                    
                    # Generate code animation
                    animation_path = self._create_code_animation(code_id, code)
                    code_animations[code_id] = animation_path
                    
                    logger.info(f"Generated code animation for section {i}")
                    
                except Exception as e:
                    logger.error(f"Code animation generation failed for section {i}: {e}")
                    raise
        
        return code_animations
    
    def _create_code_animation(self, code_id: str, code: str) -> str:
        """
        Create an animated code walkthrough video.
        
        Args:
            code_id: ID for the code section
            code: Python code to animate
            
        Returns:
            Path to the generated video file
        """
        # Create a unique filename based on a hash of the code
        code_hash = str(hash(code))[:10]
        animation_path = os.path.join("code_animations", f"code_{code_hash}.mp4")
        
        # If animation already exists, return it
        if os.path.exists(animation_path):
            return animation_path
            
        # Split code into lines
        code_lines = code.strip().split("\n")
        
        clips = []
        duration_per_line = 2.5  # Increased from 1.5 to 2.5 seconds per line
        
        # Define syntax highlighting colors
        colors = {
            'keyword': (86, 156, 214),    # Blue for keywords
            'string': (206, 145, 120),    # Orange for strings
            'comment': (87, 166, 74),     # Green for comments
            'function': (220, 220, 170),  # Light yellow for function names
            'default': (220, 220, 220),   # Light gray for default text
            'number': (181, 206, 168),    # Light green for numbers
            'background': (30, 30, 30),   # Dark gray background
            'line_number': (128, 128, 128) # Gray for line numbers
        }
        
        # Simple syntax highlighting patterns
        patterns = {
            'keyword': r'\b(def|class|if|else|elif|for|while|return|import|from|as|try|except|finally|with|in|is|not|and|or|True|False|None)\b',
            'string': r'(\'.*?\'|\".*?\")',
            'comment': r'(#.*$)',
            'function': r'\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            'number': r'\b(\d+)\b'
        }
        
        try:
            # Try to load a nicer monospace font if available
            font_path = os.path.join("assets", "fonts", "Consolas.ttf")
            if os.path.exists(font_path):
                code_font = ImageFont.truetype(font_path, 24)
            else:
                code_font = ImageFont.load_default()
        except Exception:
            # Fall back to default font
            code_font = ImageFont.load_default()
        
        for i in range(len(code_lines)):
            # Show code up to this line
            current_code = "\n".join(code_lines[:i+1])
            
            # Create a frame using PIL
            img = Image.new('RGB', (1280, 720), color=colors['background'])
            draw = ImageDraw.Draw(img)
            
            # Add a title bar
            draw.rectangle([(0, 0), (1280, 40)], fill=(50, 50, 50))
            draw.text((640, 20), f"Python Code Example", fill=(255, 255, 255), 
                     font=code_font, anchor="mm")
            
            # Draw a border around the code area
            draw.rectangle([(40, 50), (1240, 670)], outline=(60, 60, 60), width=2)
            
            # Draw the code with line numbers and syntax highlighting
            y_position = 60
            for line_num, line in enumerate(code_lines[:i+1]):
                # Draw line number with background
                draw.rectangle([(40, y_position-5), (80, y_position+25)], fill=(40, 40, 40))
                draw.text((60, y_position), f"{line_num+1}", fill=colors['line_number'], 
                         font=code_font, anchor="mm")
                
                # Apply syntax highlighting
                x_position = 100
                remaining_line = line
                
                # Check for comments first (they take precedence)
                comment_match = re.search(patterns['comment'], line)
                if comment_match:
                    comment_start = comment_match.start()
                    # Draw the part before the comment
                    if comment_start > 0:
                        pre_comment = line[:comment_start]
                        draw.text((x_position, y_position), pre_comment, fill=colors['default'], font=code_font)
                        x_position += draw.textlength(pre_comment, font=code_font)
                    
                    # Draw the comment
                    comment = line[comment_start:]
                    draw.text((x_position, y_position), comment, fill=colors['comment'], font=code_font)
                else:
                    # No comment, process the whole line
                    while remaining_line:
                        matched = False
                        
                        # Try to match each pattern
                        for pattern_name, pattern in patterns.items():
                            if pattern_name == 'comment':
                                continue  # Already handled comments
                                
                            match = re.search(f'^{pattern}', remaining_line)
                            if match:
                                matched_text = match.group(0)
                                draw.text((x_position, y_position), matched_text, 
                                         fill=colors[pattern_name], font=code_font)
                                x_position += draw.textlength(matched_text, font=code_font)
                                remaining_line = remaining_line[len(matched_text):]
                                matched = True
                                break
                        
                        # If no pattern matched, draw the next character in default color
                        if not matched:
                            char = remaining_line[0]
                            draw.text((x_position, y_position), char, fill=colors['default'], font=code_font)
                            x_position += draw.textlength(char, font=code_font)
                            remaining_line = remaining_line[1:]
                
                y_position += 30
                
                # Highlight the current line (the last line being added)
                if line_num == i:
                    draw.rectangle([(40, y_position-30), (1240, y_position)], outline=(100, 100, 255), width=1)
            
            # Save the frame to a temporary file
            os.makedirs("temp", exist_ok=True)
            frame_path = os.path.join("temp", f"code_frame_{i}.png")
            img.save(frame_path)
            
            # Create a clip from the frame
            frame_clip = ImageClip(frame_path).set_duration(duration_per_line if i < len(code_lines) - 1 else 5.0)  # Increased final frame duration from 3 to 5 seconds
            clips.append(frame_clip)
        
        # Concatenate all clips
        final_clip = concatenate_videoclips(clips)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(animation_path), exist_ok=True)
        
        # Write the video file
        final_clip.write_videofile(animation_path, fps=24, codec='libx264')
        
        # Clean up temporary frame files
        for i in range(len(code_lines)):
            frame_path = os.path.join("temp", f"code_frame_{i}.png")
            if os.path.exists(frame_path):
                os.remove(frame_path)
        
        return animation_path
    
    def assemble_video(
        self, 
        script_data: Dict, 
        audio_files: Dict[str, str], 
        slide_files: Dict[str, str], 
        code_animations: Dict[str, str]
    ) -> str:
        """
        Assemble the final tutorial video.
        
        Args:
            script_data: The structured script data
            audio_files: Mapping of section IDs to audio files
            slide_files: Mapping of section IDs to slide images
            code_animations: Mapping of section IDs to code animation videos
            
        Returns:
            Path to the final video file
        """
        logger.info("Assembling final video")
        
        # Create video segments for each section
        video_segments = []
        
        # Add title segment
        title_segment = self._create_title_segment(
            slide_files["title"], audio_files["intro"]
        )
        video_segments.append(title_segment)
        
        # Add intro segment
        intro_segment = self._create_slide_segment(
            slide_files["intro"], audio_files["intro"]
        )
        video_segments.append(intro_segment)
        
        # Add section segments
        for i, section in enumerate(script_data["sections"]):
            section_id = f"section_{i}"
            
            # Add section explanation segment
            section_segment = self._create_slide_segment(
                slide_files[section_id], audio_files[section_id]
            )
            video_segments.append(section_segment)
            
            # Add code demonstration if available
            code_id = f"code_{i}"
            if code_id in code_animations and code_id in audio_files:
                code_segment = self._create_code_segment(
                    code_animations[code_id], audio_files[code_id]
                )
                video_segments.append(code_segment)
        
        # Add conclusion segment
        conclusion_segment = self._create_slide_segment(
            slide_files["conclusion"], audio_files["conclusion"]
        )
        video_segments.append(conclusion_segment)
        
        # Concatenate all segments
        final_video = concatenate_videoclips(video_segments)
        
        # Add background music if specified
        if "music" in self.config and self.config["music"]["enabled"]:
            final_video = self._add_background_music(final_video)
        
        # Write the final video
        tutorial_name = script_data["title"].replace(" ", "_").lower()
        output_path = os.path.join("output", f"{tutorial_name}.mp4")
        final_video.write_videofile(output_path, fps=24, codec='libx264')
        
        logger.info(f"Video assembled and saved to {output_path}")
        return output_path
    
    def _create_title_segment(self, title_image: str, audio_file: str) -> VideoFileClip:
        """
        Create the title segment of the video.
        
        Args:
            title_image: Path to the title slide image
            audio_file: Path to the intro audio
            
        Returns:
            VideoFileClip of the title segment
        """
        # Create image clip from the title slide
        image_clip = ImageClip(title_image)
        
        # Load audio and set the duration
        audio = AudioFileClip(audio_file)
        duration = 5.0  # Fixed duration for title
        
        # Set the duration of the image clip
        image_clip = image_clip.set_duration(duration)
        
        # Set the audio for the first part of the audio
        image_clip = image_clip.set_audio(audio.subclip(0, duration))
        
        return image_clip
    
    def _create_slide_segment(self, slide_image: str, audio_file: str) -> VideoFileClip:
        """
        Create a segment with a slide and narration.
        
        Args:
            slide_image: Path to the slide image
            audio_file: Path to the narration audio
            
        Returns:
            VideoFileClip of the slide segment
        """
        # Create image clip from the slide
        image_clip = ImageClip(slide_image)
        
        # Load audio
        audio = AudioFileClip(audio_file)
        
        # Set the duration of the image clip to match the audio
        image_clip = image_clip.set_duration(audio.duration)
        
        # Set the audio
        image_clip = image_clip.set_audio(audio)
        
        return image_clip
    
    def _create_code_segment(self, code_video: str, audio_file: str) -> VideoFileClip:
        """
        Create a segment with code animation and narration.
        
        Args:
            code_video: Path to the code animation video
            audio_file: Path to the narration audio
            
        Returns:
            VideoFileClip of the code segment
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Load the code animation video
            video_clip = VideoFileClip(code_video)
            
            # Load audio
            audio = AudioFileClip(audio_file)
            
            # Add a short pause at the beginning to let viewers prepare for the code demonstration
            pause_duration = 1.0  # 1 second pause
            
            # Create a freeze frame at the beginning of the video
            first_frame = video_clip.to_ImageClip(0)
            first_frame = first_frame.set_duration(pause_duration)
            
            # Create a freeze frame at the end to allow time for explanation
            end_pause_duration = 3.0  # 3 second pause at the end
            last_frame = video_clip.to_ImageClip(video_clip.duration - 0.1)  # Get frame just before end
            last_frame = last_frame.set_duration(end_pause_duration)
            
            # Concatenate the freeze frames with the original video
            extended_video = concatenate_videoclips([first_frame, video_clip, last_frame])
            
            # Determine the final duration (max of video and audio)
            final_duration = max(extended_video.duration, audio.duration)
            
            # If video is shorter than audio, extend it with the last frame
            if extended_video.duration < audio.duration:
                try:
                    # Create an additional freeze frame to match audio duration
                    extra_freeze_duration = audio.duration - extended_video.duration
                    extra_last_frame = last_frame.set_duration(extra_freeze_duration)
                    
                    # Concatenate with the extra freeze frame
                    extended_video = concatenate_videoclips([extended_video, extra_last_frame])
                except Exception as e:
                    logger.warning(f"Could not create additional freeze frame: {str(e)}. Using existing video duration.")
            
            # If audio is shorter than video, extend it with silence
            if audio.duration < extended_video.duration:
                silence = AudioClip(lambda t: 0, duration=extended_video.duration - audio.duration)
                audio = CompositeAudioClip([audio, silence.set_start(audio.duration)])
            
            # Set the audio
            extended_video = extended_video.set_audio(audio)
            
            return extended_video
            
        except Exception as e:
            logger.error(f"Error creating code segment from {code_video}: {str(e)}")
            
            # Create a fallback clip with just the audio
            width, height = self.config["video"]["width"], self.config["video"]["height"]
            
            # Create a black background
            fallback_clip = ColorClip(size=(width, height), color=(0, 0, 0), duration=10)
            
            # Add text explaining the issue
            text_clip = TextClip(
                "Code animation could not be loaded.\nContinuing with tutorial...",
                fontsize=30,
                color='white',
                bg_color='black',
                font=self.config["fonts"]["main"]
            ).set_position('center').set_duration(10)
            
            # Composite the clips
            fallback_clip = CompositeVideoClip([fallback_clip, text_clip])
            
            # Add the audio if possible
            try:
                audio = AudioFileClip(audio_file)
                fallback_clip = fallback_clip.set_audio(audio)
                
                # Adjust duration to match audio if needed
                if audio.duration > fallback_clip.duration:
                    fallback_clip = fallback_clip.set_duration(audio.duration)
            except Exception:
                # If audio fails too, just use the fallback clip as is
                pass
                
            return fallback_clip
    
    def _add_background_music(self, video: VideoFileClip) -> VideoFileClip:
        """
        Add background music to the video.
        
        Args:
            video: The final video clip
            
        Returns:
            VideoFileClip with background music added
        """
        # Load the background music
        music_path = self.config["music"]["file"]
        music = AudioFileClip(music_path)
        
        # Loop the music if needed to match video duration
        if music.duration < video.duration:
            repeated_music = []
            current_time = 0
            
            while current_time < video.duration:
                clip_duration = min(music.duration, video.duration - current_time)
                music_segment = music.subclip(0, clip_duration)
                repeated_music.append(music_segment.set_start(current_time))
                current_time += clip_duration
            
            music = CompositeAudioClip(repeated_music)
        else:
            music = music.subclip(0, video.duration)
        
        # Adjust the volume
        music = music.volumex(self.config["music"]["volume"])
        
        # Mix with the original audio
        final_audio = CompositeAudioClip([video.audio, music])
        video = video.set_audio(final_audio)
        
        return video
    
    def add_monetization_elements(self, video_path: str, script_data: Dict) -> str:
        """
        Add monetization elements to the video (ad breaks, sponsorship segment).
        
        Args:
            video_path: Path to the assembled video
            script_data: The structured script data
            
        Returns:
            Path to the final video with monetization elements
        """
        logger.info("Adding monetization elements")
        
        # Load the assembled video
        video = VideoFileClip(video_path)
        
        # Add sponsorship segment if enabled
        if "sponsorship" in self.config and self.config["sponsorship"]["enabled"]:
            video = self._add_sponsorship_segment(video, script_data)
        
        # Add ad break markers if using YouTube
        if "youtube" in self.config and self.config["youtube"]["ad_breaks"]:
            # For YouTube, you would set ad break points in the YouTube Studio dashboard
            # after upload. Here we just note the optimal break points.
            optimal_ad_breaks = self._calculate_optimal_ad_breaks(video.duration)
            logger.info(f"Optimal ad break points: {optimal_ad_breaks}")
        
        # Write the final video with monetization
        monetized_path = video_path.replace(".mp4", "_monetized.mp4")
        video.write_videofile(monetized_path, fps=24, codec='libx264')
        
        return monetized_path
    
    def _add_sponsorship_segment(self, video: VideoFileClip, script_data: Dict) -> VideoFileClip:
        """
        Add a sponsorship segment to the video.
        
        Args:
            video: The video clip
            script_data: The structured script data
            
        Returns:
            VideoFileClip with sponsorship segment added
        """
        # Generate the sponsorship segment
        sponsor_name = self.config["sponsorship"]["name"]
        sponsor_script = self.config["sponsorship"]["script"]
        
        # Create TTS for the sponsorship
        sponsor_audio_path = self._call_tts_api("sponsorship", sponsor_script)
        sponsor_audio = AudioFileClip(sponsor_audio_path)
        
        # Create a visual for the sponsorship
        sponsor_visual_path = self._generate_sponsorship_slide(sponsor_name)
        sponsor_image = ImageClip(sponsor_visual_path).set_duration(sponsor_audio.duration)
        sponsor_image = sponsor_image.set_audio(sponsor_audio)
        
        # Determine where to insert the sponsorship (e.g., 1/3 into the video)
        insert_point = video.duration / 3
        
        # Split the video and insert the sponsorship
        first_part = video.subclip(0, insert_point)
        second_part = video.subclip(insert_point)
        
        # Concatenate with the sponsorship in the middle
        final_video = concatenate_videoclips([first_part, sponsor_image, second_part])
        
        return final_video
    
    def _calculate_optimal_ad_breaks(self, video_duration: float) -> List[float]:
        """
        Calculate optimal points for ad breaks.
        
        Args:
            video_duration: Total duration of the video in seconds
            
        Returns:
            List of timestamps (in seconds) for optimal ad breaks
        """
        # For videos over 8 minutes, YouTube allows mid-roll ads
        if video_duration < 480:  # 8 minutes
            return []
        
        ad_breaks = []
        
        # First ad break after the intro (about 2 minutes in)
        ad_breaks.append(120)
        
        # If video is long enough, add another break around 2/3 of the way through
        if video_duration > 600:  # 10 minutes
            ad_breaks.append(video_duration * 2 / 3)
        
        return ad_breaks

def main():
    """Main entry point for the tutorial generator."""
    parser = argparse.ArgumentParser(description="Generate Python tutorial videos")
    parser.add_argument("topic", help="Python topic to create a tutorial for")
    parser.add_argument(
        "--difficulty",
        choices=["beginner", "intermediate", "advanced"],
        default="intermediate",
        help="Target audience skill level"
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--tts-provider",
        choices=["elevenlabs", "gtts", "pyttsx3"],
        help="Text-to-speech provider to use (overrides config file setting)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override TTS provider if specified
        if args.tts_provider:
            config["tts"]["provider"] = args.tts_provider
            print(f"Using TTS provider: {args.tts_provider}")
        
        # Create generator
        generator = TutorialGenerator(config)
        
        # Generate tutorial
        output_path = generator.generate_tutorial(args.topic, args.difficulty)
        
        print(f"Tutorial generated successfully: {output_path}")
        
    except Exception as e:
        logger.exception("Tutorial generation failed")
        print(f"Error: {e}")
        return 1
    
    return 0

# Example config.json structure
DEFAULT_CONFIG = {
    "openai": {
        "api_key": "your-openai-api-key",
        "model": "o1-mini"
    },
    "tts": {
        "provider": "elevenlabs",
        "api_key": "your-elevenlabs-api-key",
        "voice_id": "your-voice-id"
    },
    "music": {
        "enabled": True,
        "file": "path-to-music-file.mp3",
        "volume": 0.1
    },
    "sponsorship": {
        "enabled": False,
        "name": "SponsorName",
        "script": "This tutorial is brought to you by SponsorName, the best platform for learning Python."
    },
    "youtube": {
        "ad_breaks": True
    }
}

if __name__ == "__main__":
    exit(main())
