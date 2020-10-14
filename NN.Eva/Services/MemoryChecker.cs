using System;
using System.IO;
using NN.Eva.Core;
using NN.Eva.Models;

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
        /// <param name="netStructure"></param>
        /// <returns>Validity of memory</returns>
        public bool IsValid(string memoryPath, NetworkStructure networkStructure)
        {
            string[] splittedPath = memoryPath.Split(new string[] {"//"}, StringSplitOptions.RemoveEmptyEntries);
            string memoryFolderPath = splittedPath[0];
            string memoryFileName = splittedPath[splittedPath.Length - 1];

            // Если файл не поврежден при прошлой записи, запуск полной проверки в соответствии со структурой:
            if (IsValidQuickCheck(memoryFolderPath, memoryPath, networkStructure))
            {
                FileManager fileManager = new FileManager(memoryFolderPath);

                NeuralNetwork testNet = new NeuralNetwork(networkStructure.NeuronsByLayers, fileManager);

                if(!testNet.IsMemoryEquals(networkStructure))
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
    }
}
