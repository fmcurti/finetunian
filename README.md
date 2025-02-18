# finetunIan

finetunIan is a project aimed at fine-tuning OpenAI's Whisper model to improve ASR accuracy for a single speaker, specifically targeting "Talon Voice".
The model obtains improved results on voice command with as little as 100 samples, without sacrificing the original accuracy of the model.

## Project Structure

The project consists of two main components:

1. **Audio Labeler**: A Streamlit application for recording audio samples from predefined texts
2. **Training Pipeline**: Scripts for fine-tuning and evaluating Whisper models using the recorded audio dataset

### Audio Labeler

The labeler component is a Streamlit app that allows recording audio samples for a predefined set of texts including prose passages and voice commands.

To run the labeler:

```bash
streamlit run labeler/record_dataset.py
```

This will start a web interface where you can:

* View text prompts to record
* Record audio samples
* Navigate through the dataset
* Upload recordings to create your training data

### Training and Evaluation

#### Training
```bash
python train.py --model "openai/whisper-small" \
                --data_dir "dataset" \
                --output_dir "models/whisper-small-finetunian" \
                --batch_size 2 \
                --max_steps 150
```
#### Evaluating
```bash
python metrics.py --model "models/whisper-small-finetunian/checkpoint-200" \
                  --batch_size 2 \
                  --test_samples 200
```

### Requirements
To install dependencies run
```bash
pip install -r requirements.txt
```