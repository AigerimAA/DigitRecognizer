import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import sys


def load_session(model_path):
    return ort.InferenceSession(model_path)


def get_image_files(folder):
    return [f for f in os.listdir(folder) if f.endswith('.jpg')]


def image_to_tensor(image_path):
    img = Image.open(image_path).convert('L').resize((28, 28), Image.Resampling.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 1, 28, 28)


def predict(session, image_path):
    tensor = image_to_tensor(image_path)
    output = session.run(None, {"Input3": tensor})
    return int(np.argmax(output[0]))


def count_digits(session, folder, files):
    counts = [0] * 10

    for i, filename in enumerate(files):
        path = os.path.join(folder, filename)
        digit = predict(session, path)
        counts[digit] += 1

        if (i + 1) % 500 == 0:
            print(f"Processed: {i + 1}/{len(files)}")

    return counts


def print_results(counts):
    print(f"\nResult:")
    print(f"{counts}")
    print(f"Sum: {sum(counts)}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python check.py <imagesFolder> <modelPath>")
        return

    images_folder = sys.argv[1]
    model_path = sys.argv[2]

    session = load_session(model_path)
    files = get_image_files(images_folder)

    print(f"Files found: {len(files)}")

    counts = count_digits(session, images_folder, files)
    print_results(counts)


if __name__ == "__main__":
    main()