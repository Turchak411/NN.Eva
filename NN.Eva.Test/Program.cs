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
                InputVectorLength = 2,
                NeuronsByLayers = new[] { 23, 15, 12, 1 }
            };

            TrainingConfiguration trainConfig = new TrainingConfiguration
            {
                TrainingAlgorithmType = TrainingAlgorithmType.RProp,
                StartIteration = 0,
                EndIteration = 1000,
                InputDatasetFilename = "TrainingSets//inputSets.txt",
                OutputDatasetFilename = "TrainingSets//outputSets.txt",
                MemoryFolder = "Memory"
            };

            bool creatingSucceed = serviceEvaNN.CreateNetwork(trainConfig.MemoryFolder, netStructure);

            if (creatingSucceed)
            {
                //serviceEvaNN.CalculateStatistic(trainConfig);
                serviceEvaNN.Train(trainConfig,
                                   true,
                                   ProcessPriorityClass.Normal,
                                   true);
                //serviceEvaNN.CalculateStatistic(trainConfig);
            }

            Console.WriteLine("Done!");
            Console.ReadKey();
        }
    }
}
