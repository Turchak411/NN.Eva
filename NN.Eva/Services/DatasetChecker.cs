using NN.Eva.Models;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;

namespace NN.Eva.Services
{
    public class DatasetChecker
    {
        public bool CheckInputDataset(ref string errorMessage, string datasetFilepath, NetworkStructure networkStructure)
        {
            List<double[]> dataset = FileManager.LoadTrainingDataset(datasetFilepath);

            if (dataset.Count == 0)
            {
                errorMessage += "Input dataset is empty!\n";
                return false;
            }

            for(int i = 0; i < dataset.Count; i++)
            {
                if (dataset[i].Length != networkStructure.InputVectorLength)
                {
                    errorMessage += $"-----------------------------------------\n" +
                                    $"Input dataset's rows does not match existent network's structure!\n" +
                                    $"Network structure requires: { networkStructure.InputVectorLength } rows.\n" +
                                    $"Input dataset has: { dataset[i].Length } rows.\n" +
                                    $"-----------------------------------------\n";
                    return false;
                }
            }

            errorMessage += "";
            return true;
        }

        public bool CheckOutputDataset(ref string errorMessage, string datasetFilepath, NetworkStructure networkStructure)
        {
            List<double[]> dataset = FileManager.LoadTrainingDataset(datasetFilepath);

            if (dataset.Count == 0)
            {
                errorMessage += "Output dataset is empty!\n";
                return false;
            }

            for (int i = 0; i < dataset.Count; i++)
            {
                if (dataset[i].Length != networkStructure.NeuronsByLayers[networkStructure.NeuronsByLayers.Length - 1])
                {
                    errorMessage += $"-----------------------------------------\n" +
                                    $"Output dataset's rows does not match existent network's structure!\n" +
                                    $"Network structure requires: { networkStructure.NeuronsByLayers[networkStructure.NeuronsByLayers.Length - 1] } rows.\n" +
                                    $"Output dataset has: { dataset[i].Length } rows.\n" +
                                    $"-----------------------------------------\n";
                    return false;
                }
            }

            errorMessage += "no_errors";
            return true;
        }

        public bool CheckTrainingSetsCounts(ref string errorMessage, string inputSetFilepath, string outputSetFilepath)
        {
            List<double[]> inputSet = FileManager.LoadTrainingDataset(inputSetFilepath);
            List<double[]> outputSet = FileManager.LoadTrainingDataset(outputSetFilepath);

            if (inputSet.Count != outputSet.Count)
            {
                errorMessage += $"-----------------------------------------\n" +
                                $"Training dataset's rows count does not equals!\n" +
                                $"Output dataset has: { inputSet.Count } rows.\n" +
                                $"Output dataset has: { outputSet.Count } rows.\n" +
                                $"-----------------------------------------\n";
                return false;
            }

            errorMessage += "no_errors";
            return true;
        }

        public void DoSimilarityGraphicReport(string inputSetFilepath, string filePath)
        {
            // Calculate all distances for input set:
            List<double[]> inputSet = FileManager.LoadTrainingDataset(inputSetFilepath);
            List<double> avgDistances = new List<double>();

            for(int i = 0; i < inputSet.Count; i++)
            {
                double avgDistance = 5.0;

                for (int k = 0; k < inputSet.Count; k++)
                {
                    if (i != k)
                    {
                        avgDistance += GetSimilarity(inputSet[i], inputSet[k]);
                    }
                }

                avgDistances.Add(avgDistance / (inputSet.Count - 1));
            }

            // Define bottom line (value) for graphic:
            double botValue = avgDistances.Max();

            // Drawing result image:
            DrawReportImage(filePath, avgDistances, botValue, 75);
        }

        private double GetSimilarity(double[] set0, double[] set1)
        {
            double upValue = 0;
            double downValueLeft = 0;
            double downValueRight = 0;

            for (int i = 0; i < set0.Length; i++)
            {
                upValue += set0[i] * set1[i];
                downValueLeft += set0[i] * set0[i] + 1e-16;
                downValueRight += set1[i] * set1[i] + 1e-16;
            }

            return upValue / (Math.Sqrt(downValueLeft) * Math.Sqrt(downValueRight));
        }

        private void DrawReportImage(string imgPath, List<double> avgDistances, double botValue, int imgHeight = 75)
        {
            // Create empty image:
            Bitmap bitmapImg = new Bitmap(avgDistances.Count, imgHeight);

            // Setting pixels:
            for (int x = 0; x < bitmapImg.Width; x++)
            {
                double percent = avgDistances[x] / botValue;

                Color columnColor = Color.FromArgb((int)(255 * (1 - percent)), (int)(255 * percent), 0);

                for (int y = 0; y < bitmapImg.Height; y++)
                {
                    bitmapImg.SetPixel(x, y, columnColor);
                }
            }

            // Saving img:
            using (MemoryStream memory = new MemoryStream())
            {
                using (FileStream fs = new FileStream(imgPath, FileMode.Create, FileAccess.ReadWrite))
                {
                    bitmapImg.Save(memory, ImageFormat.Png);
                    byte[] bytes = memory.ToArray();
                    fs.Write(bytes, 0, bytes.Length);
                }
            }
        }
    }
}
