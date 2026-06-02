using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace DigitRecognizer
{
    internal class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("Usage: DigitRecognizer <imagesFolder> <modelPath>");
                return;
            }

            var imagesFolder = args[0];
            var modelPath = args[1];

            using var session = new InferenceSession(modelPath);
            var files = Directory.GetFiles(imagesFolder, "*.jpg");

            Console.WriteLine($"Files found: {files.Length}");

            var counts = CountDigits(session, files);

            PrintResults(counts);
        }

        static int[] CountDigits(InferenceSession session, string[] files)
        {
            var counts = new int[10];

            for (int i = 0; i < files.Length; i++)
            {
                int digit = Predict(session, files[i]);
                counts[digit]++;

                if ((i + 1) % 500 == 0)
                    Console.WriteLine($"Processed: {i + 1}/{files.Length}");
            }

            return counts;
        }

        static int Predict(InferenceSession session, string imagePath)
        {
            float[] pixels = ImageToTensor(imagePath);

            var input = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("Input3", new DenseTensor<float>(pixels, new[] { 1, 1, 28, 28 }))
            };

            using var results = session.Run(input);
            float[] output = results.First().AsEnumerable<float>().ToArray();

            return Array.IndexOf(output, output.Max());
        }

        static float[] ImageToTensor(string imagePath)
        {
            using var original = SKBitmap.Decode(imagePath);
            using var resized = original.Resize(new SKImageInfo(28, 28), SKSamplingOptions.Default);

            var pixels = new float[28 * 28];

            for (int y = 0; y < 28; y++)
                for (int x = 0; x < 28; x++)
                    pixels[y * 28 + x] = GetGrayPixel(resized, x, y);

            return pixels;
        }

        static float GetGrayPixel(SKBitmap bitmap, int x, int y)
        {
            var pixel = bitmap.GetPixel(x, y);
            return (pixel.Red * 0.299f + pixel.Green * 0.587f + pixel.Blue * 0.114f) / 255f;
        }

        static void PrintResults(int[] counts)
        {
            Console.WriteLine("\nResult:");
            Console.WriteLine($"[{string.Join(", ", counts)}]");
            Console.WriteLine($"Sum: {counts.Sum()}");
        }
    }
}