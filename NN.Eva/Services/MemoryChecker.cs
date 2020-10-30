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
        /// <param name="memoryPath"></param>
        /// <param name="networkStructure"></param>
        /// <returns>Validity of memory after quick check</returns>
        public bool IsValidQuickCheck(string memoryPath, NetworkStructure networkStructure)
        {
            return IsFileNotCorrupted(memoryPath) && IsMetaDataEquals(memoryPath, networkStructure);
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
                        if (fileReader.ReadLine().Split(' ').Length < 2)
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

        private bool IsMetaDataEquals(string memoryPath, NetworkStructure networkStructure)
        {
            NetworkStructure readedNetworkStructure = ReadNetworkMetadata(memoryPath);

            return NetworkStructureModelsEquals(readedNetworkStructure, networkStructure);
        }

        /// <summary>
        /// Read the network's metadata
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public NetworkStructure ReadNetworkMetadata(string path)
        {
            using (StreamReader fileReader = new StreamReader(path))
            {
                string[] readedLine = fileReader.ReadLine().Split(' ');

                NetworkStructure networkStructure = new NetworkStructure();
                networkStructure.InputVectorLength = Int32.Parse(readedLine[0]);
                networkStructure.NeuronsByLayers = new int[readedLine.Length - 1];

                for (int i = 1; i < readedLine.Length; i++)
                {
                    networkStructure.NeuronsByLayers[i - 1] = Int32.Parse(readedLine[i]);
                }

                return networkStructure;
            }
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
            // Если файл не поврежден при прошлой записи, запуск полной проверки в соответствии со структурой:
            if (IsValidQuickCheck(memoryPath, networkStructure))
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
