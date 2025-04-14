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
import librosa
import numpy as np
import soundfile as sf
from typing import Tuple, Optional

def load_audio(file_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return audio data and sample rate.
    
    Args:
        file_path (str): Path to audio file
        sr (int, optional): Target sample rate. If None, uses original sample rate.
    
    Returns:
        Tuple[np.ndarray, int]: Audio data and sample rate
    """
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr

def save_audio(audio: np.ndarray, file_path: str, sr: int):
    """
    Save audio data to file.
    
    Args:
        audio (np.ndarray): Audio data
        file_path (str): Path to save audio file
        sr (int): Sample rate
    """
    sf.write(file_path, audio, sr)

def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract MFCC features from audio.
    
    Args:
        audio (np.ndarray): Audio data
        sr (int): Sample rate
        n_mfcc (int): Number of MFCC coefficients
    
    Returns:
        np.ndarray: MFCC features
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def extract_f0(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract F0 (fundamental frequency) and voiced flag from audio.
    
    Args:
        audio (np.ndarray): Audio data
        sr (int): Sample rate
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: F0 values and voiced flag
    """
    f0, voiced_flag = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)
    return f0, voiced_flag

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range.
    
    Args:
        audio (np.ndarray): Audio data
    
    Returns:
        np.ndarray: Normalized audio data
    """
    return librosa.util.normalize(audio) 

def extract_audio_features(file_path: str) -> np.ndarray:
    """
    Extract audio features (MFCC and F0) from audio data.
    
    Args:
        audio (np.ndarray): Audio data
        sr (int): Sample rate
    
    Returns:
        np.ndarray: Extracted features
    """
    audio, sr = load_audio(file_path)
    audio = normalize_audio(audio)
    mfcc = extract_mfcc(audio, sr)
    # f0, voiced_flag = extract_f0(audio, sr)
    
    # Combine features for training
    features = np.concatenate((mfcc), axis=1)
    
    return features