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
        image = decode_base64_image(row['image_base64'])
        
        # Generate a unique ID for the image
        unique_id = row['national_number']
        
        # Define the image file path
        image_path = os.path.join(image_subfolder, f"{unique_id}.png")
        
        # Save the image
        image.save(image_path)
        
        # Construct the answer from primary and secondary types
        primary_type = row['primary_type']
        secondary_type = row['secondary_type'] if pd.notna(row['secondary_type']) else ""
        formatted_answer = ", ".join(filter(None, [primary_type, secondary_type]))  # Only include non-empty types
        
        # Structure for LLaVA JSON
        json_data = {
            "id": unique_id,
            "image": f"{unique_id}.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "What Pokémon type is this?"
                },
                {
                    "from": "gpt",
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

def save_dataset(df, output_folder, val_split=0.2):
    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=42)
    
    # Process and save the datasets
    process_and_save(train_df, output_folder, 'train')
    process_and_save(val_df, output_folder, 'validation')

save_dataset(pokemon_df, data_dir, val_split=0.2)


### Instructions for fine-tuning LLaVA https://wandb.ai/byyoung3/ml-news/reports/How-to-Fine-Tune-LLaVA-on-a-Custom-Dataset--Vmlldzo2NjUwNTc1
### Make sure to change BASE_DIR in the creation of the sh file (echo)
# git clone https://github.com/bdytx5/finetune_LLaVA
# cd finetune_LLaVA
# git lfs install
# git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
# conda create -n llava python=3.10 -y
# conda activate llava
# pip install --upgrade pip  # enable PEP 660 support
# pip install -e .
# pip install -e ".[train]"
# pip install flash-attn --no-build-isolation
# echo -e "#!/bin/bash\n\nBASE_DIR=\"/home/raguina\"\n\ndeepspeed \$BASE_DIR/finetune_LLaVA/llava/train/train_mem.py \\\n    --deepspeed \$BASE_DIR/finetune_LLaVA/scripts/zero2.json \\\n    --lora_enable True \\\n    --lora_r 128 \\\n    --lora_alpha 256 \\\n    --mm_projector_lr 2e-5 \\\n    --bits 4 \\\n    --model_name_or_path \$BASE_DIR/finetune_LLaVA/llava/llava-v1.5-7b \\\n    --version llava_llama_2 \\\n    --data_path \$BASE_DIR/dsc190-project-proposal/real_pokemon_data/train/dataset.json \\\n    --validation_data_path \$BASE_DIR/dsc190-project-proposal/real_pokemon_data/validation/dataset.json \\\n    --image_folder \$BASE_DIR/dsc190-project-proposal/real_pokemon_data/images/ \\\n    --vision_tower openai/clip-vit-large-patch14-336 \\\n    --mm_projector_type mlp2x_gelu \\\n    --mm_vision_select_layer -2 \\\n    --mm_use_im_start_end False \\\n    --mm_use_im_patch_token False \\\n    --image_aspect_ratio pad \\\n    --group_by_modality_length True \\\n    --bf16 True \\\n    --output_dir \$BASE_DIR/finetune_LLaVA/checkpoints/llama-2-7b-chat-task-pokemon \\\n    --num_train_epochs 500 \\\n    --per_device_train_batch_size 32 \\\n    --per_device_eval_batch_size 32 \\\n    --gradient_accumulation_steps 1 \\\n    --evaluation_strategy \"epoch\" \\\n    --save_strategy \"steps\" \\\n    --save_steps 50000 \\\n    --save_total_limit 1 \\\n    --learning_rate 2e-4 \\\n    --weight_decay 0. \\\n    --warmup_ratio 0.03 \\\n    --lr_scheduler_type \"cosine\" \\\n    --logging_steps 1 \\\n    --tf32 True \\\n    --model_max_length 2048 \\\n    --gradient_checkpointing True \\\n    --dataloader_num_workers 4 \\\n    --lazy_preprocess True \\\n    --report_to wandb" > run.sh
# chmod +x run.sh
# ./run.sh
