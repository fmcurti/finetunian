import argparse
import os
import torch
from datasets import load_dataset, Audio, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from functools import partial
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

args_parser = argparse.ArgumentParser()
args_parser.add_argument("--model", type=str, default="openai/whisper-small")
args_parser.add_argument("--data_dir", type=str, default="dataset")
args_parser.add_argument("--output_dir", type=str, default="models/whisper-small-finetunian")
args_parser.add_argument("--max_steps", type=int, default=150)
args_parser.add_argument("--batch_size", type=int, default=2)
args_parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
args_parser.add_argument("--learning_rate", type=float, default=1e-6)
args_parser.add_argument("--warmup_steps", type=int, default=50)
args_parser.add_argument("--fp16", type=bool, default=True)
args_parser.add_argument("--save_steps", type=int, default=50)
args_parser.add_argument("--eval_steps", type=int, default=50)
args_parser.add_argument("--logging_steps", type=int, default=25)


# Load and prepare dataset
def load_and_prepare_dataset(model, data_dir):
    finetunian = load_dataset('audiofolder', data_dir=data_dir)
    
    processor = WhisperProcessor.from_pretrained(
        model, language="english", task="transcribe"
    )
    
    # Resample audio to 16kHz
    sampling_rate = processor.feature_extractor.sampling_rate
    finetunian = finetunian.cast_column("audio", Audio(sampling_rate=sampling_rate))
    
    # Prepare dataset
    def prepare_dataset(example):
        audio = example["audio"]
        example = processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            text=example["transcription"],
        )
        example["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        return example
    
    # Map and filter dataset
    finetunian = finetunian.map(
        prepare_dataset, 
        remove_columns=finetunian.column_names["train"], 
        num_proc=1
    )
    
    # Filter long audio files
    max_input_length = 30.0
    def is_audio_in_length_range(length):
        return length < max_input_length
    
    finetunian["train"] = finetunian["train"].filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
    )
    
    return finetunian, processor

# Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def get_metrics_function(processor):
    metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Compute orthographic WER
        wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

        # Compute normalized WER
        pred_str_norm = [normalizer(pred) for pred in pred_str]
        label_str_norm = [normalizer(label) for label in label_str]
        
        # Filter empty references
        pred_str_norm = [
            pred_str_norm[i] for i in range(len(pred_str_norm)) 
            if len(label_str_norm[i]) > 0
        ]
        label_str_norm = [
            label_str_norm[i] for i in range(len(label_str_norm))
            if len(label_str_norm[i]) > 0
        ]

        wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)
        return {"wer_ortho": wer_ortho, "wer": wer}
    
    return compute_metrics

def main():
    # Parse arguments
    args = args_parser.parse_args()
    print(args)
    # Load and prepare dataset
    finetunian, processor = load_and_prepare_dataset(args.model, args.data_dir)
    
    # Initialize model
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.use_cache = False
    
    # Set generation parameters
    model.generate = partial(
        model.generate, language="english", task="transcribe", use_cache=True
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=args.fp16,
        fp16_full_eval=True,
        eval_strategy="steps",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=finetunian["train"],
        eval_dataset=finetunian["train"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=get_metrics_function(processor),
        processing_class=processor,
    )
    
    # Train
    trainer.train()
    
if __name__ == "__main__":
    main()