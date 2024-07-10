import os
from PIL import Image, ImageDraw
import cv2
import numpy as np
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch
import evaluate
import torchvision.transforms as transforms

# Configuración de rutas
CSV_PATH = 'labels.csv'
IMAGE_DIR = 'images'
MODEL_SAVE_PATH = './trained_model'
IMAGE_PATH = 'EVAL3.jpg'

def load_and_preprocess_data():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    df = pd.read_csv(CSV_PATH)

    transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomHorizontalFlip(),
    ])

    def augment_image(image_path):
        image = Image.open(os.path.join(IMAGE_DIR, image_path)).convert("RGB")
        augmented_images = [transform(image) for _ in range(10)]  # Genera 10 variaciones
        return augmented_images

    def preprocess_function(examples):
        all_images = []
        all_texts = []
        for img_path, text in zip(examples['image_path'], examples['text']):
            augmented_images = augment_image(img_path)
            all_images.extend(augmented_images)
            all_texts.extend([text] * len(augmented_images))
        
        inputs = processor(images=all_images, return_tensors='pt', padding=True, truncation=True)
        labels = processor(text=all_texts, return_tensors='pt', padding=True, truncation=True)
        return {
            'pixel_values': inputs.pixel_values,
            'labels': labels.input_ids
        }

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(preprocess_function, batched=True, remove_columns=["image_path", "text"])
    dataset = dataset.shuffle(seed=42)
    dataset_split = dataset.train_test_split(test_size=0.2)
    dataset_dict = DatasetDict({
        'train': dataset_split['train'],
        'validation': dataset_split['test']
    })

    dataset_dict.save_to_disk('saved_dataset')
    return dataset_dict, processor

def preprocess_image(image_path):
    # Abrir la imagen
    image = cv2.imread(image_path)

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral para binarizar
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Suponiendo que el texto de "Total" es uno de los contornos más grandes
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Encontrar la caja delimitadora más probable
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > image.shape[1] * 0.5 and h < image.shape[0] * 0.1:  # Condiciones basadas en proporciones típicas
            cropped = image[y:y+h, x:x+w]
            return Image.fromarray(cropped)
    
    # Si no se encuentra una región adecuada, devolver la imagen original
    return Image.fromarray(image)

def extract_text_trocr(image_path):
    processor = TrOCRProcessor.from_pretrained(MODEL_SAVE_PATH)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_SAVE_PATH)

    image = preprocess_image(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values, max_length=100, num_beams=8, early_stopping=True)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("Texto generado:", generated_text)
    return generated_text

def train_model():
    dataset, processor = load_and_preprocess_data()
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    model.config.decoder_start_token_id = processor.tokenizer.pad_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=10,
        save_total_limit=2,
        lr_scheduler_type='linear',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=processor
    )

    trainer.train()
    model.save_pretrained(MODEL_SAVE_PATH)
    processor.save_pretrained(MODEL_SAVE_PATH)

def evaluate_model():
    dataset = DatasetDict.load_from_disk('saved_dataset')
    processor = TrOCRProcessor.from_pretrained(MODEL_SAVE_PATH)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_SAVE_PATH)

    test_dataset = dataset['validation']

    predictions = []
    labels = [processor.tokenizer.decode(example['labels'], skip_special_tokens=True) for example in test_dataset]

    for example in test_dataset:
        pixel_values = torch.tensor(example['pixel_values']).unsqueeze(0)
        generated_ids = model.generate(pixel_values, max_length=50, num_beams=8, early_stopping=True)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predictions.append(generated_text)

    metric = evaluate.load('wer', trust_remote_code=True)
    results = metric.compute(predictions=predictions, references=labels)

    print("Tipo de resultado:", type(results))
    print("Contenido del resultado:", results)

    if isinstance(results, float):
        print(f"Tasa de Error de Palabras (WER): {results}")
    else:
        print(f"Tasa de Error de Palabras (WER): {results['wer']}")

    for prediction, label in zip(predictions, labels):
        print(f"Prediction: {prediction} | Label: {label}")

if __name__ == "__main__":
    #train_model()
    #evaluate_model()
    extract_text_trocr(IMAGE_PATH)
