# SPEAK: Speech Processing for Enhanced Accent Kinetics

A deep learning-based system for many-to-many accent conversion, preserving speaker identity while transforming accent characteristics.

## Project Overview

SPEAK is designed to effectively disentangle and manipulate accent characteristics of speech without affecting the speaker's unique vocal attributes or the intended message. The system addresses both phonetic variations and prosodic features that differ across accents.

## Features

- Many-to-many accent conversion
- Speaker identity preservation
- Phonetic and prosodic feature manipulation
- Real-time processing capabilities
- Comprehensive evaluation framework

## Project Structure

```
SPEAK/
├── config/                 # Configuration files
├── datasets/               # Dataset management
├── scripts/                # Model architectures and implementations
├── evaluation/            # Evaluation metrics and tools
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/RowanRama/SPEAK.git
cd SPEAK
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate speak
```

3. Install additional dependencies (if needed):
```bash
pip install -r requirements.txt
```

## Usage

[Usage instructions will be added as the project develops]

## Datasets

The project utilizes the following datasets:
- Speech Accent Archive
- L2-ARCTIC corpus
- LJSpeech dataset
- LibriTTS

## Evaluation Metrics

- Speaker Similarity (SSIM, Cosine Similarity)
- Accent Accuracy (Perceptual Assessment)
- Phonetic Accuracy (WER, PER)
- Prosody Analysis (F0 contours, duration metrics)
- Human Subjective Evaluation (MOS)

## License

MIT License

Copyright (c) 2025 Efthymios Marios Loukedes, Rowan Ramamurthy, Umut Zengin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Authors

- Efthymios Marios Loukedes
- Rowan Ramamurthy
- Umut Zengin