import random
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from config.env
load_dotenv('config.env')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Get the OpenAI API key from the environment

# List of Pokémon types
pokemon_types = [
    "fire", "water", "grass", "electric", "psychic", 
    "rock", "ground", "flying", "ice", "dragon"
]

# Function to generate a description for a given Pokémon type
def generate_pokemon_description(pokemon_type, few_shot=False):
    # Zero-shot prompt
    zero_shot_prompt = f"Create a vivid and detailed description of a {pokemon_type}-type Pokémon. Include details like its appearance, abilities, and habitat."

    # Few-shot example (if needed)
    example = """
    Entry example: This {pokemon_type} Pokemon is small, and cute mouse-like Pokémon. They are almost completely covered by yellow fur. They have long yellow ears that are tipped with black. A Pikachu's back has two brown stripes, and its large tail is notable for being shaped like a lightning bolt, yet its brown tip is almost always forgotten. Pikachu have short arms with five tiny fingers on forehands and three sharp fingers on their hind legs. On its cheeks are two red, circle-shaped pouches used for storing its electricity. 
    Now, generate a similar description for a {pokemon_type}-type Pokémon.
    """
    few_shot_prompt = example.replace("{pokemon_type}", pokemon_type)

    # Use few-shot or zero-shot based on the input parameter
    prompt = few_shot_prompt if few_shot else zero_shot_prompt

    # Call GPT-4 with the prompt and return the description
    description = call_gpt_4(prompt)
    return description

# Function to call GPT-4 API using ChatCompletion
def call_gpt_4(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates Pokémon descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.8,
            n=1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling GPT-4 API: {e}")
        return "No description generated due to an API error."

# Main pipeline to generate descriptions for a set number of Pokémon types
def generate_pokemon_descriptions(num_pokemon=1, few_shot=False):
    descriptions = []

    # Select random types if num_pokemon is less than available types
    selected_types = random.sample(pokemon_types, min(num_pokemon, len(pokemon_types)))

    # Generate descriptions for each type
    for pokemon_type in selected_types:
        description = generate_pokemon_description(pokemon_type, few_shot=few_shot)
        descriptions.append({
            "type": pokemon_type,
            "description": description
        })
        print(f"Generated description for {pokemon_type}-type Pokémon: {description}\n")

    return descriptions

# Example usage
if __name__ == "__main__":
    num_pokemon = 1  # Number of Pokémon descriptions to generate
    few_shot = True  # Use few-shot prompting for more detailed descriptions
    descriptions = generate_pokemon_descriptions(num_pokemon, few_shot)

    # Output the generated descriptions (you could also save to a file if needed)
    for entry in descriptions:
        print(f"Type: {entry['type']}")
        print(f"Description: {entry['description']}\n")
