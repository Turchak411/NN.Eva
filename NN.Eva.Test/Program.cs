using System;
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
                NeuronsByLayers = new[] { 230, 180, 160, 80, 1 },
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
                serviceEvaNN.CalculateStatistic(trainConfig);
                //serviceEvaNN.Train(trainConfig,
                //                   true,
                //                   ProcessPriorityClass.Normal,
                //                   true);
                //serviceEvaNN.CalculateStatistic(trainConfig);
            }

            Console.WriteLine("Done!");
            Console.ReadKey();
        }
    }
}
