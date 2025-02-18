import torch
from datasets import load_dataset, Dataset, Audio, DatasetDict
from transformers import pipeline
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from evaluate import load
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ASR model metrics')
    parser.add_argument('--model', type=str, 
                       default='models/whisper-base-finetunian/checkpoint-200',
                       help='Path to model or model identifier from huggingface.co/models')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for inference')
    parser.add_argument('--test_samples', type=int, default=200,
                       help='Number of test samples to evaluate')
    return parser.parse_args()

# Set device
if torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
else:
    device = "cpu" 
    torch_dtype = torch.float32

def initialize_model(model_path):
    return pipeline(
        "automatic-speech-recognition",
        model=model_path,
        return_language="english",
        device=device,
    )

def get_predictions(data, asr, batch_size=2):
    all_predictions = []
    for prediction in tqdm(
        asr(
            KeyDataset(data, "audio"),
            generate_kwargs={"task": "transcribe", "language": "english"},
            batch_size=batch_size,
        ),
        total=len(data),
    ):
        all_predictions.append(prediction["text"])
    return all_predictions

def get_metrics(data, predictions, label_key='transcription'):
    wer_metric = load("wer")
    
    wer_ortho = 100 * wer_metric.compute(
        references=data[label_key], predictions=predictions
    )
    
    normalizer = BasicTextNormalizer()
    
    all_predictions_norm = [normalizer(pred) for pred in predictions]
    all_references_norm = [normalizer(label) for label in data[label_key]]
    
    all_predictions_norm = [
        all_predictions_norm[i]
        for i in range(len(all_predictions_norm))
        if len(all_references_norm[i]) > 0
    ]
    all_references_norm = [
        all_references_norm[i] 
        for i in range(len(all_references_norm))
        if len(all_references_norm[i]) > 0
    ]
    
    wer = 100 * wer_metric.compute(
        references=all_references_norm, predictions=all_predictions_norm
    )
    
    return wer, (100 - wer), wer_ortho, (100 - wer_ortho)

def evaluate_on_test(asr, batch_size=2, sample_size=200):
    common_voice = DatasetDict()
    common_voice["test"] = load_dataset(
        "mozilla-foundation/common_voice_13_0", 
        "en", 
        split="validation",
        trust_remote_code=True,
        streaming=True
    )

    subset_common_voice = {'test': []}
    for i, data in enumerate(common_voice['test']):
        if i == sample_size:
            break
        subset_common_voice['test'].append(data)

    test_data = subset_common_voice['test']
    test_predictions = get_predictions(test_data, asr, batch_size)
    test_data = Dataset.from_list(test_data)
    metrics = get_metrics(test_data, test_predictions, label_key='sentence')
    
    return metrics

if __name__ == "__main__":
    args = parse_args()
    
    # Load dataset
    dataset = load_dataset('audiofolder', data_dir='dataset')
    train_data = dataset['train']
    train_data = train_data.cast_column("audio", Audio(sampling_rate=16000))
    
    # Initialize model
    asr = initialize_model(args.model)
    
    train_predictions = get_predictions(train_data, asr, args.batch_size)
    train_metrics = get_metrics(train_data, train_predictions)
    test_metrics = evaluate_on_test(asr, args.batch_size, args.test_samples)
    print("\nTraining Metrics:")
    print(f"WER: {train_metrics[0]:.2f}%")
    print(f"Word Accuracy: {train_metrics[1]:.2f}%")
    print(f"WER Ortho: {train_metrics[2]:.2f}%")
    print(f"Word Ortho Accuracy: {train_metrics[3]:.2f}%")
    

    print("\nTest Metrics:")
    print(f"WER: {test_metrics[0]:.2f}%")
    print(f"Word Accuracy: {test_metrics[1]:.2f}%")
    print(f"WER Ortho: {test_metrics[2]:.2f}%")
    print(f"Word Ortho Accuracy: {test_metrics[3]:.2f}%")