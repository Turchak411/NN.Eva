using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using NN.Eva.Models;
using NN.Eva.Services.WeightsGenerator;

namespace NN.Eva.Services
{
    public class MemoryChecker
    {
        /// <summary>
        /// Быстрая проверка памяти на отсутсвие повреждений при прошлой записи
        /// и соответствия метадаты памяти загружаемойструктуре
        /// </summary>
        /// <param name="memoryFolderPath"></param>
        /// <param name="memoryPath"></param>
        /// <param name="networkStructure"></param>
        /// <returns>Validity of memory after quick check</returns>
        public bool IsValidQuickCheck(string memoryFolderPath, string memoryPath, NetworkStructure networkStructure)
        {
            return IsFileNotCorrupted(memoryPath) && IsMetaDataEquals(memoryFolderPath, memoryPath, networkStructure);
        }

        /// <summary>
        /// Быстрая проверка памяти на отсутствие повреждений при прошлой записи
        /// </summary>
        /// <param name="memoryPath"></param>
        /// <returns>Validity of memory after quick file check</returns>
        public bool IsFileNotCorrupted(string memoryPath)
        {
            try
            {
                using (StreamReader fileReader = new StreamReader(memoryPath))
                {
                    while (!fileReader.EndOfStream)
                    {
                        string[] readedLine = fileReader.ReadLine().Split(' ');

                        if (readedLine.Length < 2)
                        {
                            return false;
                        }
                    }
                }
            }
            catch
            {
                return false;
            }

            return true;
        }

        private bool IsMetaDataEquals(string memoryFolderPath, string memoryPath, NetworkStructure networkStructure)
        {
            FileManager fileManager = new FileManager(memoryFolderPath);

            NetworkStructure readedNetworkStructure = fileManager.ReadNetworkMetadata(memoryPath);

            return NetworkStructureModelsEquals(readedNetworkStructure, networkStructure);
        }

        private bool NetworkStructureModelsEquals(NetworkStructure networkStructure0,
                                                  NetworkStructure networkStructure1)
        {
            // Checking input vector length equal:
            if (networkStructure0.InputVectorLength != networkStructure1.InputVectorLength)
            {
                return false;
            }

            // Check number of layers equal:
            if (networkStructure0.NeuronsByLayers.Length != networkStructure1.NeuronsByLayers.Length)
            {
                return false;
            }

            for (int i = 0; i < networkStructure0.NeuronsByLayers.Length; i++)
            {
                if (networkStructure0.NeuronsByLayers[i] != networkStructure1.NeuronsByLayers[i])
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Полноценная проверка файла памяти в соответствии с представленной структурой
        /// </summary>
        /// <param name="memoryPath"></param>
        /// <param name="networkStructure"></param>
        /// <returns>Validity of memory</returns>
        public bool IsValid(string memoryPath, NetworkStructure networkStructure)
        {
            string[] splittedPath = memoryPath.Split(new string[] {"//"}, StringSplitOptions.RemoveEmptyEntries);
            string memoryFolderPath = splittedPath[0];

            // Если файл не поврежден при прошлой записи, запуск полной проверки в соответствии со структурой:
            if (IsValidQuickCheck(memoryFolderPath, memoryPath, networkStructure))
            {
                ServiceWeightsGenerator serviceWeightsGenerator = new ServiceWeightsGenerator();

                List<double> memoryFromStructure = serviceWeightsGenerator.GenerateEmptyMemoryWeights(networkStructure);
                List<double> memoryFromFile = LoadWholeMemoryFile(memoryPath);

                // Проверка количества значений весов по структуре с количеством значений весов, полученной после фактической загрузки:
                if (memoryFromStructure.Count != memoryFromFile.Count)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }

            return true;
        }

        private List<double> LoadWholeMemoryFile(string fullMemoryPath)
        {
            List<double> memoryWeights = new List<double>();

            using(StreamReader fileReader = new StreamReader(fullMemoryPath))
            {
                try
                {
                    // Skip metadata:
                    fileReader.ReadLine();

                    while(!fileReader.EndOfStream)
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
    }
}
