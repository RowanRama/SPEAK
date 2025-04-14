"""
Copyright © Rowan Ramamurthy, 2025.
This software is provided "as-is," without any express or implied warranty. 
In no event shall the authors be held liable for any damages arising from 
the use of this software.

Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it 
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

Author: Rowan Ramamurthy
"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from audio_utils import save_audio#, extract_audio_features
from dataset_utils import create_accent_id
from diffusion_model import DiffusionModel, SpeechFeatureNormalizer
from model_base import BaseConversionModel
import time

# Cache configuration
CACHE_FILENAME = "data_cache.pkl"

def save_cache(data_dir, features, accents, accent_map):
    cache_path = os.path.join(data_dir, CACHE_FILENAME) 
    cache_data = {
        'features': features,
        'accents': accents,
        'accent_map': accent_map,
        'version': 1  # For future compatibility
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

def load_cache(data_dir):
    cache_path = os.path.join(data_dir, CACHE_FILENAME)
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            
        # Validate cache structure
        required_keys = {'features', 'accents', 'accent_map', 'version'}
        if not all(key in data for key in required_keys):
            raise ValueError("Invalid cache structure")
            
        if not (isinstance(data['features'], list) and
                isinstance(data['accents'], list) and
                isinstance(data['accent_map'], dict)):
            raise ValueError("Invalid data types in cache")
            
        return data['features'], data['accents'], data['accent_map']
        
    except (FileNotFoundError, pickle.UnpicklingError, ValueError) as e:
        print(f"Cache loading failed: {str(e)}")
        return None

class AccentDataset(Dataset):
    def __init__(self, features, accents, max_len=500):
        self.features = features
        self.accents = accents
        self.max_len = max_len
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feat = self.features[idx]
        # Padding/trimming to fixed length
        if len(feat) > self.max_len:
            feat = feat[:self.max_len]
        else:
            feat = np.pad(feat, ((0, self.max_len - len(feat)), (0, 0)))
            
        return {
            'features': torch.FloatTensor(feat),
            'accent': torch.LongTensor([self.accents[idx]])
        }

# Placeholder for actual feature extraction function
def extract_audio_features(filename):
    """Placeholder for actual feature extraction (e.g., mel-spectrograms)"""
    # In practice, use librosa or audio processing library
    return np.random.randn(100, 80)  # Mock features

# Load data from the accent archive
def load_accent_archive_data(data_dir, ignore_cache=False):
    """Load dataset using cache or CSV processing"""
    # Try loading cache first
    if not ignore_cache:
        cached_data = load_cache(data_dir)
        if cached_data:
            print("Using cached dataset")
            return cached_data

    # Fallback to CSV processing
    print("Processing dataset from scratch...")
    print(data_dir)
    print(os.path.abspath(data_dir))
    csv_path = os.path.join(data_dir, "speakers_all.csv")
    df = pd.read_csv(csv_path)
    
    features = []
    accents = []
    accent_map = {}
    
    for _, row in df.iterrows():
        if row['file_missing?'] == True:
            continue
            
        # Generate accent ID
        accent_id = create_accent_id(row, accent_map)
        
        # Process audio file
        audio_path = os.path.join(data_dir, "recordings", row['filename']+".mp3")
        print(audio_path)
        print(os.path.exists(audio_path))
        if not os.path.exists(audio_path):
            continue
            
        try:
            feat = extract_audio_features(str(audio_path))
            features.append(feat)
            accents.append(accent_map[accent_id])
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    
    # Save to cache
    save_cache(data_dir, features, accents, accent_map)
    
    return features, accents, accent_map


def get_model(model_choice, num_accents, feature_dim):
    if model_choice == 1:
        return DiffusionModel(num_accents=num_accents, feature_dim=feature_dim)
    # Add other models here
    else:
        raise ValueError("Invalid model choice")

def train(model, model_choice, dataloader, val_loader, normalizer, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        start_time = time.time()
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Normalize features
            features = normalizer.normalize(batch['features'].numpy())
            features = torch.FloatTensor(features).to(device)
            
            accents = batch['accent'].squeeze(1).to(device)
            
            loss = model.loss_fn(features, accents)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = normalizer.normalize(batch['features'].numpy())
                features = torch.FloatTensor(features).to(device)
                accents = batch['accent'].squeeze(1).to(device)
                
                loss = model.loss_fn(features, accents)
                val_loss += loss.item()
        
        # Logging
        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / len(dataloader)
        avg_val_loss = val_loss / len(val_loader)
        
        writer.add_scalars('Loss', {
            'train': avg_train_loss,
            'val': avg_val_loss
        }, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Time: {epoch_time:.2f}s | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), f"model_{model_choice}.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Directory containing audio files")
    parser.add_argument("-e", "--epochs", type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument("-c", "--ignore-cache", action="store_true",
                       help="Ignore existing cache and reprocess data")
    parser.add_argument("-m", "--model", type=int, choices=[1,2,3], default=1,
                       help="1: DiffusionUNet, 2-3: Other models (placeholder)")
    args = parser.parse_args()

    # Load and preprocess data
    features, accents, accent_map = load_accent_archive_data(args.data_dir, args.ignore_cache)
    print(f"Loaded {len(features)} samples with {len(accent_map)} accents")
    print("Sample accents:", list(accent_map.keys())[:])
    
    # # Create normalizer
    # normalizer = SpeechFeatureNormalizer()
    # all_features = np.concatenate(features)
    # normalizer.fit(all_features)
    
    # # Create datasets
    # dataset = AccentDataset(features, accents)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # # split the data better
    # train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=16)
    
    # # Initialize model
    # feature_dim = features[0].shape[1]
    # model = get_model(args.model, len(accent_map), feature_dim)
    
    # # Train
    # train(model, args.model, train_loader, val_loader, normalizer)
