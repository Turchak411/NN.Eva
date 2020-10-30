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

        /// <summary>
        /// Result of success of network creating
        /// </summary>
        public bool IsMemoryLoadCorrect;

        private Logger _logger;

        /// <summary>
        /// Main use
        /// </summary>
        /// <param name="netStructure"></param>
        /// <param name="memoryFolderPath"></param>
        /// <param name="defaultMemoryFilePath"></param>
        public FileManager(NetworkStructure netStructure = null, string memoryFolderPath = "Memory", string defaultMemoryFilePath = "memoryClear.txt")
        {
            if (memoryFolderPath == "" || defaultMemoryFilePath == "")
            {
                IsMemoryLoadCorrect = false;
                return;
            }

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

        /// <summary>
        /// Checking memory for equaling to checked
        /// </summary>
        /// <param name="memoryPathToCheck"></param>
        /// <returns></returns>
        public bool IsMemoryEqualsDefault(string memoryPathToCheck)
        {
            FileInfo fileDefaultMemory = new FileInfo(MemoryFolderPath + "//.clear//" + DefaultMemoryFilePath);
            FileInfo fileToCheck = new FileInfo(MemoryFolderPath + "//" + memoryPathToCheck);

            // Возвращает false, если вес проверяемого файла памяти отличается от файла с чистой памятью больше чем на 50%
            return Math.Abs(fileDefaultMemory.Length - fileToCheck.Length) < fileDefaultMemory.Length * 0.5;
        }

        /// <summary>
        /// Loading network's clear memory
        /// </summary>
        /// <param name="layerNumber"></param>
        /// <param name="neuronNumber"></param>
        /// <param name="offsetValue"></param>
        /// <param name="offsetWeight"></param>
        /// <returns></returns>
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
                        offsetValue = double.Parse(readedLine[2], new CultureInfo("ru-RU"));
                        offsetWeight = double.Parse(readedLine[3], new CultureInfo("ru-RU"));
                        memory = GetWeights(readedLine);
                        break;
                    }
                }
            }

            return memory;
        }

        /// <summary>
        /// Loading network's exists memory from memory-file
        /// </summary>
        /// <param name="layerNumber"></param>
        /// <param name="neuronNumber"></param>
        /// <param name="memoryPath"></param>
        /// <param name="offsetValue"></param>
        /// <param name="offsetWeight"></param>
        /// <returns></returns>
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
                        offsetValue = double.Parse(readedLine[2], new CultureInfo("ru-RU"));
                        offsetWeight = double.Parse(readedLine[3], new CultureInfo("ru-RU"));
                        memory = GetWeights(readedLine);
                        break;
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

        /// <summary>
        /// Used in saving method for preparing memory files to saving procedure
        /// </summary>
        /// <param name="path"></param>
        /// <param name="networkStructure"></param>
        public void PrepareToSaveMemory(string path, NetworkStructure networkStructure)
        {
            try
            {
                File.Delete(path);
            }
            catch { }

            WriteNetworkMetadata(networkStructure, path);
        }

        private void WriteNetworkMetadata(NetworkStructure networkStructure, string path)
        {
            // Запись мета данных в начало файла памяти:
            using (StreamWriter fileWriter = new StreamWriter(path))
            {
                fileWriter.Write(networkStructure.InputVectorLength);

                for (int i = 0; i < networkStructure.NeuronsByLayers.Length; i++)
                {
                    fileWriter.Write(" " + networkStructure.NeuronsByLayers[i]);
                }

                fileWriter.WriteLine();
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
                fileWriter.Write("layer_{0} neuron_{1} {2} {3}", layerNumber, neuronNumber, 
                                                                 offsetValue.ToString().Replace('.', ','),
                                                                 offsetWeight.ToString().Replace('.', ','));

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
        public void SaveMemoryFromModel(NetworkStructure networkStructure, List<string> memoryInTextList, string destinationMemoryFilePath)
        {
            WriteNetworkMetadata(networkStructure, destinationMemoryFilePath);

            using(StreamWriter fileWriter = new StreamWriter(destinationMemoryFilePath, true))
            {
                for (int i = 0; i < memoryInTextList.Count; i++)
                {
                    fileWriter.WriteLine(memoryInTextList[i]);
                }
            }
        }

        /// <summary>
        /// Saving memory from network's weights and structure
        /// </summary>
        /// <param name="weights"></param>
        /// <param name="networkStructure"></param>
        public void SaveMemoryFromWeightsAndStructure(List<double> weights, NetworkStructure networkStructure)
        {
            WriteNetworkMetadata(networkStructure, MemoryFolderPath + "//memory.txt");
            int index = 0;

            double offsetValue = 0.5;
            double offsetWeight = -1;

            using (StreamWriter fileWriter = new StreamWriter(MemoryFolderPath + "//memory.txt", true))
            {
                // Save the first layer:
                for (int i = 0; i < networkStructure.NeuronsByLayers[0]; i++)
                {
                    fileWriter.Write("layer_0 neuron_{0} {1} {2}",
                                                        i,
                                                        offsetValue.ToString().Replace('.', ','),
                                                        offsetWeight.ToString().Replace('.', ','));

                    for (int k = 0; k < networkStructure.InputVectorLength; k++)
                    {
                        fileWriter.Write(" " + weights[index].ToString().Replace('.', ','));
                        index++;
                    }

                    fileWriter.WriteLine();
                }

                // Save the other layers:
                for (int i = 1; i < networkStructure.NeuronsByLayers.Length; i++)
                {
                    for (int k = 0; k < networkStructure.NeuronsByLayers[i]; k++)
                    {
                        fileWriter.Write("layer_{0} neuron_{1} {2} {3}",
                                                                        i,
                                                                        k,
                                                                        offsetValue.ToString().Replace('.', ','),
                                                                        offsetWeight.ToString().Replace('.', ','));

                        for (int j = 0; j < networkStructure.NeuronsByLayers[i - 1]; j++)
                        {
                            fileWriter.Write(" " + weights[index].ToString().Replace('.', ','));
                            index++;
                        }

                        fileWriter.WriteLine();
                    }
                }
            }
        }

        public List<double> LoadWholeMemoryFile(string fullMemoryPath)
        {
            List<double> memoryWeights = new List<double>();

            using (StreamReader fileReader = new StreamReader(fullMemoryPath))
            {
                try
                {
                    // Skip metadata:
                    fileReader.ReadLine();

                    while (!fileReader.EndOfStream)
                    {
                        string[] readedLine = fileReader.ReadLine().Split(' ');

                        for (int i = 4; i < readedLine.Length; i++)
                        {
                            memoryWeights.Add(double.Parse(readedLine[i], new CultureInfo("ru-RU")));
                        }
                    }
                }
                catch
                {
                    return new List<double>();
                }
            }

            return memoryWeights;
        }

        /// <summary>
        /// Loading testing dataset
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
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
                        inputVector[i] = double.Parse(readedData[i + 1], CultureInfo.GetCultureInfo("ru-RU"));
                    }

                    vectors.Add(new TrainObject(readedData[0], inputVector));
                }
            }

            return vectors;
        }

        /// <summary>
        /// Loading training dataset
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
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
                        set[i] = double.Parse(readedLine[i], CultureInfo.GetCultureInfo("ru-RU"));
                    }

                    sets.Add(set);
                }
            }

            return sets;
        }
    }
}
