#!/bin/bash
# Examples of how to use the Python Tutorial Generator

# Example 1: Generate a tutorial on Python decorators
python tutorial_generator.py "Python decorators" --difficulty intermediate

# Example 2: Generate a beginner tutorial on list comprehensions
python tutorial_generator.py "List comprehensions in Python" --difficulty beginner

# Example 3: Generate an advanced tutorial on asynchronous programming
python tutorial_generator.py "Asynchronous programming with asyncio" --difficulty advanced

# Example 4: Using a custom configuration file
python tutorial_generator.py "Python virtual environments" --config custom_config.json

# Example 5: Using Docker
docker build -t tutorial-generator .
docker run -v $(pwd)/config.json:/app/config.json \
           -v $(pwd)/output:/app/output \
           -e OPENAI_API_KEY=your-openai-api-key \
           -e ELEVENLABS_API_KEY=your-elevenlabs-api-key \
           tutorial-generator "Python decorators" --difficulty intermediate

# Example 6: Using Docker Compose
export OPENAI_API_KEY=your-openai-api-key
export ELEVENLABS_API_KEY=your-elevenlabs-api-key
export TOPIC="Python generators"
export DIFFICULTY=intermediate
docker-compose up

# Example 7: Batch processing multiple tutorials
topics=(
  "Python data types"
  "Functions in Python"
  "Object-oriented programming in Python"
  "Python exception handling"
  "Working with files in Python"
)

for topic in "${topics[@]}"; do
  echo "Generating tutorial for: $topic"
  python tutorial_generator.py "$topic" --difficulty intermediate
  sleep 10  # Add delay to avoid rate limits
done
