using System;
using System.Collections.Generic;
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
                InputVectorLength = 2,
                NeuronsByLayers = new[] { 10, 1 },
                Alpha = 1
            };

            TrainingConfiguration trainConfig = new TrainingConfiguration
            {
                TrainingAlgorithmType = TrainingAlgorithmType.BProp,
                StartIteration = 0,
                EndIteration = 1_000_000,
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
                ////serviceEvaNN.CheckDatasetsVectorsSimilarity(trainConfig.InputDatasetFilename);
            }

            Console.WriteLine("Done!");
            Console.ReadKey();
        }
    }
}
