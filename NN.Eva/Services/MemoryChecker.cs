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
        /// </summary>
        /// <param name="memoryPath"></param>
        /// <returns></returns>
        public bool IsValidQuickCheck(string memoryPath)
        {
            try
            {
                using (StreamReader fileReader = new StreamReader(memoryPath))
                {
                    while (!fileReader.EndOfStream)
                    {
                        string[] readedLine = fileReader.ReadLine().Split(' ');

                        if (readedLine.Length < 3)
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

        /// <summary>
        /// Полноценная проверка файла памяти в соответствии с представленной структурой
        /// </summary>
        /// <param name="memoryPath"></param>
        /// <param name="netStructure"></param>
        /// <returns></returns>
        public bool IsValid(string memoryPath, NetworkStructure netStructure)
        {
            string[] splittedPath = memoryPath.Split(new string[] {"//"}, StringSplitOptions.RemoveEmptyEntries);
            string memoryFolderPath = splittedPath[0];
            string memoryFileName = splittedPath[splittedPath.Length - 1];

            // Если файл не поврежден при прошлой записи, запуск полной проверки в соответствии со структурой:
            if (IsValidQuickCheck(memoryPath))
            {
                FileManager fileManager = new FileManager(memoryFolderPath);

                NeuralNetwork testNet = new NeuralNetwork(netStructure.InputVectorLength,
                                                          netStructure.NeuronsByLayers,
                                                          fileManager);

                if(!testNet.IsMemoryEquals(netStructure))
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
