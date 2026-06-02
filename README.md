# Digit Recognizer

A console app that counts how many times each digit (0-9) appears in 12000 digit images.

## How it works

1. Loads a pre-trained MNIST model in ONNX format
2. For each image: resizes it to 28x28 pixels and converts to grayscale
3. Passes the image through the model
4. The model predicts which digit is drawn
5. Counts predictions for each digit 

## Tech stack

- C# / .NET
- Microsoft.ML.OnnxRuntime
- SkiaSharp
- MNIST model

## Result
[1271, 1393, 950, 1299, 1203, 1294, 1205, 940, 970, 1475]

Total files processed: 12000

## Verification

Result was also verified using Python with the same ONNX model (check.py)
