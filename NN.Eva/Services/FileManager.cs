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
        public FileManager(string memoryFolderPath = "Memory", string defaultMemoryFilePath = "memoryClear.txt")
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
        public FileManager(NetworkStructure netStructure = null, string memoryFolderPath = "Memory", string defaultMemoryFilePath = "memoryClear.txt")
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

        public bool IsMemoryEqualsDefault(string memoryPathToCheck)
        {
            FileInfo fileDefaultMemory = new FileInfo(MemoryFolderPath + "//.clear//" + DefaultMemoryFilePath);
            FileInfo fileToCheck = new FileInfo(MemoryFolderPath + "//" + memoryPathToCheck);

            // Возвращает false, если вес проверяемого файла памяти отличается от файла с чистой памятью больше чем на 50%
            return Math.Abs(fileDefaultMemory.Length - fileToCheck.Length) < fileDefaultMemory.Length * 0.5;
        }

        public double[] LoadMemory(int layerNumber, int neuronNumber, ref double offsetValue, ref double offsetWeight)
        {
            double[] memory = new double[0];

            using (StreamReader fileReader = new StreamReader(MemoryFolderPath + "//.clear//" + DefaultMemoryFilePath))
            {
                while (!fileReader.EndOfStream)
                {
                    string[] readedLine = fileReader.ReadLine().Split(' ');

                    if ((readedLine[0] == "layer_" + layerNumber) && (readedLine[1] == "neuron_" + neuronNumber))
                    {
                        // TODO: Еще протестировать на других настройках осей
                        offsetValue = double.Parse(readedLine[2].Replace('.', ','));
                        offsetWeight = double.Parse(readedLine[3].Replace('.', ','));
                        memory = GetWeights(readedLine);
                        break;
                    }
                }
            }

            return memory;
        }

        public double[] LoadMemory(int layerNumber, int neuronNumber, string memoryPath, ref double offsetValue, ref double offsetWeight)
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
                        // TODO: Еще протестировать на других настройках осей
                        offsetValue = double.Parse(readedLine[2].Replace('.', ','));
                        offsetWeight = double.Parse(readedLine[3].Replace('.', ','));
                        memory = GetWeights(readedLine);
                    }
                }
            }

            return memory;
        }

        private double[] GetWeights(string[] readedLine)
        {
            double[] weights = new double[readedLine.Length - 4];

            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = double.Parse(readedLine[i + 4], CultureInfo.GetCultureInfo("ru-RU"));
            }

            return weights;
        }

        public void PrepareToSaveMemory(string path, NetworkStructure networkStructure)
        {
            try
            {
                File.Delete(path);
            }
            catch { }

            // Запись мета данных в начало файла памяти:
            using (StreamWriter fileWriter = new StreamWriter(path))
            {
                fileWriter.Write(networkStructure.InputVectorLength);

                for(int i = 0; i < networkStructure.NeuronsByLayers.Length; i++)
                {
                    fileWriter.Write(" " + networkStructure.NeuronsByLayers[i]);
                }

                fileWriter.WriteLine();
            }
        }

        public NetworkStructure ReadNetworkMetadata(string path)
        {
            using (StreamReader fileReader = new StreamReader(path))
            {
                string[] readedLine = fileReader.ReadLine().Split(' ');

                NetworkStructure networkStructure = new NetworkStructure();
                networkStructure.InputVectorLength = Int32.Parse(readedLine[0]);
                networkStructure.NeuronsByLayers = new int[readedLine.Length - 1];

                for(int i = 1; i < readedLine.Length; i++)
                {
                    networkStructure.NeuronsByLayers[i - 1] = Int32.Parse(readedLine[i]);
                }

                return networkStructure;
            }
        }

        /// <summary>
        /// Main method of saving memory
        /// </summary>
        /// <param name="layerNumber"></param>
        /// <param name="neuronNumber"></param>
        /// <param name="weights"></param>
        /// <param name="path"></param>
        public void SaveMemory(int layerNumber, int neuronNumber, double[] weights, double offsetValue, double offsetWeight, string path)
        {
            using (StreamWriter fileWriter = new StreamWriter(path, true))
            {
                fileWriter.Write("layer_{0} neuron_{1} {2} {3}", layerNumber, neuronNumber, offsetValue, offsetWeight);

                for (int i = 0; i < weights.Length; i++)
                {
                    fileWriter.Write(" " + weights[i].ToString().Replace('.',','));
                }

                fileWriter.WriteLine("");
            }
        }

        /// <summary>
        /// Saving memory from textList memory-model
        /// </summary>
        /// <param name="memoryInTextList"></param>
        public void SaveMemoryFromModel(List<string> memoryInTextList, string destinationMemoryFilePath)
        {
            using(StreamWriter fileWriter = new StreamWriter(destinationMemoryFilePath))
            {
                for (int i = 0; i < memoryInTextList.Count; i++)
                {
                    fileWriter.WriteLine(memoryInTextList[i]);
                }
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
                    string[] readedLine = fileReader.ReadLine().Trim().Split(' ');
                    double[] set = new double[readedLine.Length];

                    for (int i = 0; i < readedLine.Length; i++)
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
