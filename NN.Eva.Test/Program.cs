using System;
using System.Diagnostics;
using NN.Eva.Models;

namespace NN.Eva.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            ServiceEvaNN serviceEvaNN = new ServiceEvaNN();

            NetworkStructure netStructure = new NetworkStructure
            {
                InputVectorLength = 10,
                NeuronsByLayers = new[] { 1337, 1 }
            };

            TrainingConfiguration trainConfig = new TrainingConfiguration
            {
                StartIteration = 0,
                EndIteration = 120000,
                InputDatasetFilename = "TrainingSets//inputSets.txt",
                OutputDatasetFilename = "TrainingSets//outputSets.txt",
                MemoryFolder = "Memory"
            };

            bool creatingSucceed = serviceEvaNN.CreateNetwork(trainConfig.MemoryFolder, netStructure);

            if (creatingSucceed)
            {
                serviceEvaNN.Train(trainConfig,
                                   true,
                                   ProcessPriorityClass.High,
                                   true);
                //serviceEvaNN.CalculateStatistic(trainConfig);
            }

            //DatabaseConfig dbConfig = new DatabaseConfig
            //{
            //    Server = "localhost",
            //    Database = "memorynnn",
            //    Password = "root",
            //    UID = "root"
            //};

            //serviceEvaNN.BackupMemory("Memory", dbConfig, "first backup :)");

            Console.WriteLine("Done!");
            Console.ReadKey();
        }
    }
}
