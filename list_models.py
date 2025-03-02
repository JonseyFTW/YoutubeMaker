import os
import json
import openai

# Load config
def load_config(config_path="config.json"):
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        raise

# Main function
def main():
    # Load configuration
    config = load_config()
    
    # Set OpenAI API key
    openai.api_key = config["openai"]["api_key"]
    
    try:
        # List available models
        models = openai.Model.list()
        print("Available models:")
        for model in models.data:
            print(f"- {model.id}")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    main()
