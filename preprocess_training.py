import os
import json
import uuid
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split

data_dir = 'real_pokemon_data'
pokemon_df = pd.read_csv(os.path.join(data_dir, 'processed_data.csv'))
pokemon_df.head()

def decode_base64_image(base64_string):
    """Decodes a base64 image and returns a PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

from PIL import Image

def process_and_save(df, output_folder, subset_name):
    # Create subfolders for images and JSON files
    subset_folder = os.path.join(output_folder, subset_name)
    image_subfolder = os.path.join(output_folder, 'images')
    
    os.makedirs(image_subfolder, exist_ok=True)
    os.makedirs(subset_folder, exist_ok=True)
    
    # Initialize list to store JSON data
    json_data_list = []
    
    for _, row in df.iterrows():
        # Decode the base64 image
        # image = decode_base64_image(row['image_base64'])
        
        # # Convert P or RGBA to RGB if necessary
        # if image.mode in ['P', 'RGBA']:
        #     image = image.convert('RGB')
        
        # Generate a unique ID for the image
        unique_id = row['national_number']
        
        # # Define the image file path
        image_path = os.path.join(image_subfolder, f"{unique_id}.jpg")  # Save as JPG
        
        # # Save the image
        # image.save(image_path, format='JPEG')  # Ensure format is JPEG
        
        # Construct the answer from primary and secondary types
        primary_type = row['primary_type']
        # secondary_type = row['secondary_type'] if pd.notna(row['secondary_type']) else ""
        # formatted_answer = ", ".join(filter(None, [primary_type, secondary_type]))  # Only include non-empty types
        formatted_answer = primary_type
        # Structure for JSON
        json_data = {
            "id": unique_id,
            "image": f"{unique_id}.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "What Pokémon type is this?"
                },
                {
                    "from": "gt",
                    "value": formatted_answer
                }
            ]
        }
        print(f'image saved to {image_path}')
        # Append JSON data to the list
        json_data_list.append(json_data)
    
    # Save JSON data to a file
    json_output_path = os.path.join(subset_folder, 'dataset.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)

def save_dataset(df, output_folder, train_split=0.2, val_split=0.1, test_split=0.7):
    """
    Splits the dataset into training, validation, and testing sets and processes them.
    
    Args:
        df (pd.DataFrame): The original dataframe.
        output_folder (str): Path to save the output datasets.
        train_split (float): Proportion of the original data for the training set.
        val_split (float): Proportion of the original data for the validation set.
        test_split (float): Proportion of the original data for the testing set.
    
    Note:
        The train_split + val_split + test_split must equal 1.0.
    """
    if not np.isclose(train_split + val_split + test_split, 1.0):
        raise ValueError("The sum of train_split, val_split, and test_split must equal 1.0.")
    
    # First, split off the training set from the main dataset
    train_df, remaining_df = train_test_split(df, test_size=(1 - train_split), random_state=42)
    
    # Then, split the remaining data into validation and test sets
    # Normalize val_split and test_split to the remaining proportion
    remaining_val_split = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(remaining_df, test_size=(1 - remaining_val_split), random_state=42)
    
    # Process and save the datasets
    process_and_save(train_df, output_folder, 'train')
    process_and_save(val_df, output_folder, 'validation')
    process_and_save(test_df, output_folder, 'test')

# Save the dataset with specified validation and testing splits
save_dataset(pokemon_df, data_dir, train_split=0.2, val_split=0.1, test_split=0.7)