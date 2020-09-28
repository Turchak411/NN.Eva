using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using NN.Eva.Models;
using NN.Eva.Services.WeightsGenerator;

namespace NN.Eva.Services
{
    public class FileManager
    {
        /// <summary>
        /// Path of default memory
        /// </summary>
        public string DefaultMemoryFilePath;

        /// <summary>
        /// Path of a memory folder
        /// </summary>
        public string MemoryFolderPath;

        public bool IsMemoryLoadCorrect;

        private Logger _logger;

        /// <summary>
        /// Uses only for checking memory validity
        /// </summary>
        /// <param name="memoryFolderPath"></param>
        public FileManager(string memoryFolderPath = "Memory", string defaultMemoryFilePath = "memory.txt")
        {
            IsMemoryLoadCorrect = true;

            _logger = new Logger();

            DefaultMemoryFilePath = defaultMemoryFilePath;
            MemoryFolderPath = memoryFolderPath;

            // Check for existing memory folder:
            if (!Directory.Exists(MemoryFolderPath))
            {
                Directory.CreateDirectory(MemoryFolderPath);
            }

            // Запуск процесса генерации памяти в случае ее отсутствия:
            if (!File.Exists(MemoryFolderPath + "//.clear//" + DefaultMemoryFilePath))
            {
                _logger.LogError(ErrorType.MemoryMissing);

                Directory.CreateDirectory(MemoryFolderPath + "//.clear");

                Console.WriteLine("Start generating process...");
                ServiceWeightsGenerator weightsGenerator = new ServiceWeightsGenerator();

                weightsGenerator.GenerateMemory(MemoryFolderPath + "//.clear//" + DefaultMemoryFilePath);
            }
        }

        /// <summary>
        /// Main use
        /// </summary>
        /// <param name="netStructure"></param>
        /// <param name="memoryFolderPath"></param>
        /// <param name="defaultMemoryFilePath"></param>
        public FileManager(NetworkStructure netStructure = null, string memoryFolderPath = "Memory", string defaultMemoryFilePath = "memory.txt")
        {
            IsMemoryLoadCorrect = true;

            _logger = new Logger();

            DefaultMemoryFilePath = defaultMemoryFilePath;
            MemoryFolderPath = memoryFolderPath;

            // Check for existing memory folder:
            if (!Directory.Exists(MemoryFolderPath))
            {
                Directory.CreateDirectory(MemoryFolderPath);
            }

            // Запуск процесса генерации памяти в случае ее отсутствия:
            if (!File.Exists(MemoryFolderPath + "//.clear//" + DefaultMemoryFilePath))
            {
                _logger.LogError(ErrorType.MemoryMissing);

                Directory.CreateDirectory(MemoryFolderPath + "//.clear");

                Console.WriteLine("Start generating process...");
                ServiceWeightsGenerator weightsGenerator = new ServiceWeightsGenerator();

                if (netStructure != null)
                {
                    weightsGenerator.GenerateMemory(MemoryFolderPath + "//.clear//" + DefaultMemoryFilePath, netStructure.InputVectorLength, netStructure.NeuronsByLayers);
                }
                else
                {
                    weightsGenerator.GenerateMemory(MemoryFolderPath + "//.clear//" + DefaultMemoryFilePath);
                }
            }
            else
            {
                // Дополнительная проверка, если файл памяти существует, но не подходит по структуре
                MemoryChecker memoryChecker = new MemoryChecker();
                
                if (!memoryChecker.IsValid(MemoryFolderPath + "//.clear//" + DefaultMemoryFilePath, netStructure))
                {
                    IsMemoryLoadCorrect = false;
                    _logger.LogError(ErrorType.MemoryInitializeError);
                }
            }
        }

        public double[] LoadMemory(int layerNumber, int neuronNumber)
        {
            double[] memory = new double[0];

            using (StreamReader fileReader = new StreamReader(MemoryFolderPath + "//.clear//" + DefaultMemoryFilePath))
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

            // Создание памяти для отдельного класса в случае отсутствия таковой:
            if (!File.Exists(MemoryFolderPath + "//" + memoryPath))
            {
                File.Copy(MemoryFolderPath + "//.clear//" + DefaultMemoryFilePath, MemoryFolderPath + "//" + memoryPath);
            }

            // Загрузка весов нейрона:
            using (StreamReader fileReader = new StreamReader(MemoryFolderPath + "//" + memoryPath))
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
                weights[i] = double.Parse(readedLine[i + 2].Replace('.', ','));
            }

            return weights;
        }

        public void PrepareToSaveMemory(string path)
        {
            File.Delete(path);
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

        public List<TrainObject> LoadTestDataset(string filePath)
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
                    string[] readedData = fileReader.ReadLine().Split(' ');

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

        public List<double[]> LoadTrainingDataset(string path)
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
