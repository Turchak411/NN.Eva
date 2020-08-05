using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace NN.Eva.Services
{
    public class FileManager
    {
        public string DataPath;

        private Logger _logger;

        private FileManager() { }

        public FileManager(NetworkStructure netStructure = null, string dataPath = "memory.txt")
        {
            _logger = new Logger();

            DataPath = dataPath;

            // Запуск процесса генерации памяти в случае ее отсутствия:
            if (!File.Exists(dataPath))
            {
                _logger.LogError(ErrorType.MemoryMissing);

                Console.WriteLine("Start generating process...");
                ServiceWeightsGenerator weightsGenerator = new ServiceWeightsGenerator();

                if (netStructure != null)
                {
                    weightsGenerator.GenerateMemory(DataPath, netStructure.InputVectorLength, netStructure.NeuronsByLayers);
                }
                else
                {
                    weightsGenerator.GenerateMemory(DataPath);
                }
            }
        }

        public double[] LoadMemory(int layerNumber, int neuronNumber)
        {
            double[] memory = new double[0];

            using (StreamReader fileReader = new StreamReader(DataPath))
            {
                while (!fileReader.EndOfStream)
                {
                    string[] readedLine = fileReader.ReadLine().Split(' ');

                    if ((readedLine[0] == "layer_" + layerNumber) && (readedLine[1] == "neuron_" + neuronNumber))
                    {
                        memory = GetWeights(readedLine);
                    }
                }
            }

            return memory;
        }

        public double[] LoadMemory(int layerNumber, int neuronNumber, string memoryPath)
        {
            double[] memory = new double[0];

            if (!File.Exists(memoryPath))
            {
                // Создание памяти для отдельного класса в случае отсутствия таковой
                File.Copy(DataPath, memoryPath);
            }

            using (StreamReader fileReader = new StreamReader(memoryPath))
            {
                while (!fileReader.EndOfStream)
                {
                    string[] readedLine = fileReader.ReadLine().Split(' ');

                    if ((readedLine[0] == "layer_" + layerNumber) && (readedLine[1] == "neuron_" + neuronNumber))
                    {
                        memory = GetWeights(readedLine);
                    }
                }
            }

            return memory;
        }

        private double[] GetWeights(string[] readedLine)
        {
            double[] weights = new double[readedLine.Length - 2];

            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = double.Parse(readedLine[i + 2]);
            }

            return weights;
        }

        public void PrepareToSaveMemory()
        {
            File.Delete(DataPath);
        }

        public void PrepareToSaveMemory(string path)
        {
            File.Delete(path);
        }

        public void SaveMemory(int layerNumber, int neuronNumber, double[] weights)
        {
            using (StreamWriter fileWriter = new StreamWriter(DataPath, true))
            {
                fileWriter.Write("layer_{0} neuron_{1}", layerNumber, neuronNumber);

                for (int i = 0; i < weights.Length; i++)
                {
                    fileWriter.Write(" " + weights[i]);
                }

                fileWriter.WriteLine("");
            }
        }

        public void SaveMemory(int layerNumber, int neuronNumber, double[] weights, string path)
        {
            using (StreamWriter fileWriter = new StreamWriter(path, true))
            {
                fileWriter.Write("layer_{0} neuron_{1}", layerNumber, neuronNumber);

                for (int i = 0; i < weights.Length; i++)
                {
                    fileWriter.Write(" " + weights[i]);
                }

                fileWriter.WriteLine("");
            }
        }

        public List<TrainObject> LoadDatasets(string filePath)
        {
            if (!File.Exists(filePath))
            {
                _logger.LogError(ErrorType.SetMissing, filePath + "is missing!");
                return null;
            }

            var vectors = new List<TrainObject>();
            using (StreamReader fileReader = new StreamReader(filePath, Encoding.Default))
            {
                while (!fileReader.EndOfStream)
                {
                    var readedData = fileReader.ReadLine().Split(' ');

                    // Check for space in the last position:
                    int additionalSpaceIndex = readedData[readedData.Length - 1] == "" ? 1 : 0;

                    var inputVector = new double[readedData.Length - 1 - additionalSpaceIndex];

                    for (int i = 0; i < readedData.Length - 1 - additionalSpaceIndex; i++)
                    {
                        inputVector[i] = double.Parse(readedData[i + 1]);
                    }

                    vectors.Add(new TrainObject(readedData[0], inputVector));
                }
            }

            return vectors;
        }

        public List<double[]> LoadSingleDataset(string path)
        {
            List<double[]> sets = new List<double[]>();

            using (StreamReader fileReader = new StreamReader(path))
            {
                while (!fileReader.EndOfStream)
                {
                    string[] readedLine = fileReader.ReadLine().Split(' ');
                    double[] set = new double[readedLine.Length - 1];

                    for (int i = 0; i < readedLine.Length - 1; i++)
                    {
                        set[i] = double.Parse(readedLine[i]);
                    }

                    sets.Add(set);
                }
            }

            return sets;
        }
    }
}
