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
                NeuronsByLayers = new[] { 170, 160, 140, 70, 2 },
                Alpha = 5
            };

            TrainingConfiguration trainConfig = new TrainingConfiguration
            {
                TrainingAlgorithmType = TrainingAlgorithmType.RProp,
                StartIteration = 0,
                EndIteration = 10,
                InputDatasetFilename = "TrainingSets//inputSets.txt",
                OutputDatasetFilename = "TrainingSets//outputSets.txt",
                MemoryFolder = "Memory"
            };

            bool creatingSucceed = serviceEvaNN.CreateNetwork(trainConfig.MemoryFolder, netStructure);

            if (creatingSucceed)
            {
                //serviceEvaNN.CalculateStatistic(trainConfig);
                //serviceEvaNN.Train(trainConfig,
                //                   true,
                //                   ProcessPriorityClass.Normal,
                //                   true);
                serviceEvaNN.CheckDatasetsVectorsSimilarity(trainConfig.InputDatasetFilename);
                //serviceEvaNN.CalculateStatistic(trainConfig);
            }

            Console.WriteLine("Done!");
            Console.ReadKey();
        }
    }
}
