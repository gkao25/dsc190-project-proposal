import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split

IMAGE_FOLDER = 'images'

rand_img = False

data_dir = 'real_pokemon_data'
pokemon_df = pd.read_csv(os.path.join(data_dir, 'processed_data.csv'))
pokemon_df['is_synthetic'] = False

n_pokemon = pokemon_df['national_number'].max()
synthetic_df = pd.read_csv('best_pair_v2.csv')
synthetic_df = synthetic_df.rename(columns={'Type': 'primary_type', 'Description': 'description', 'Image Path': 'path'})
synthetic_df['is_synthetic'] = True
synthetic_df['path'] = synthetic_df['path'].apply(lambda x: ('/'.join(x.split('/')[1:])))
if rand_img:
    synthetic_df['path'] = synthetic_df['path'].apply(lambda x: x[:-5] + '1.png')
synthetic_df['national_number'] = np.arange(n_pokemon, synthetic_df.shape[0] + n_pokemon)

# Combine datasets but keep track of synthetic data
combined_dataset = pd.concat([pokemon_df, synthetic_df])
combined_dataset = combined_dataset[combined_dataset['secondary_type'].isna()]

import base64
from io import BytesIO
from PIL import Image

def decode_base64_image(base64_string):
    """Decodes a base64 image and returns a PIL Image."""
    # Ensure the input is a string
    if isinstance(base64_string, float):
        print(base64_string, 'WHATTT')
        base64_string = str(base64_string)
    # Decode the base64 string
    image_data = base64.b64decode(base64_string)
    # Return the PIL Image
    return Image.open(BytesIO(image_data))



def process_and_save(df, output_folder, subset_name, use_image_path=False):
    # Create subfolders for JSON files
    subset_folder = os.path.join(output_folder, subset_name)
    image_subfolder = os.path.join(output_folder, IMAGE_FOLDER)
    
    os.makedirs(image_subfolder, exist_ok=True)
    os.makedirs(subset_folder, exist_ok=True)
    
    # Initialize list to store JSON data
    json_data_list = []
    
    for _, row in df.iterrows():
        # If synthetic, use path directly
        image_path = row['path'] if row['is_synthetic'] else f"{row['national_number']}.jpg"
        folder_path = os.path.join(image_subfolder, image_path)
        if not os.path.exists(folder_path):
            # Decode the base64 image
            image = decode_base64_image(row['image_base64'])
            if image.mode in ['P', 'RGBA']:
                image = image.convert('RGB')
            image.save(folder_path, format='JPEG')  # Ensure format is JPEG
            print(f'image saved to {folder_path}')
        
        # Construct the answer from primary and secondary types
        primary_type = row['primary_type']
        secondary_type = row['secondary_type'] if pd.notna(row['secondary_type']) else None
        formatted_answer = primary_type
        desc = row['description']
        
        # Structure for JSON
        json_data = {
            "id": row['national_number'],
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "What Pokémon type is this?"
                },
                {
                    "from": "gt",
                    "value": formatted_answer,
                    "value2": secondary_type,
                    "desc": desc,
                }
            ]
        }
        
        # Append JSON data to the list
        json_data_list.append(json_data)
    
    # Save JSON data to a file
    json_output_path = os.path.join(subset_folder, 'dataset.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)


def save_dataset(df, output_folder, train_split=0.7, val_split=0.1, test_split=0.2):
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
    
    # Separate synthetic and original data
    synthetic_data = df[df['is_synthetic']]
    original_data = df[~df['is_synthetic']]
    
    # Split original data into train, validation, and test sets
    train_df, remaining_df = train_test_split(original_data, test_size=(1 - train_split), random_state=42)
    remaining_val_split = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(remaining_df, test_size=(1 - remaining_val_split), random_state=42)
    
    n_synthetic = synthetic_data.shape[0]
    
    # Combine synthetic data with training set
    train_df = pd.concat([train_df, synthetic_data.sample(int(n_synthetic * 1))])
    
    # Process and save datasets
    process_and_save(train_df, output_folder, 'train')
    process_and_save(val_df, output_folder, 'validation')
    process_and_save(test_df, output_folder, 'test')


# Save the dataset with specified validation and testing splits
save_dataset(combined_dataset, data_dir, train_split=0.7, val_split=0.1, test_split=0.2)