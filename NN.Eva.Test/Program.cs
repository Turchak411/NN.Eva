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
                NeuronsByLayers = new[] { 230, 150, 120, 1 }
            };

            TrainingConfiguration trainConfig = new TrainingConfiguration
            {
                StartIteration = 8000,
                EndIteration = 8030,
                InputDatasetFilename = "TrainingSets//inputSets.txt",
                OutputDatasetFilename = "TrainingSets//outputSets.txt",
                MemoryFolder = "Memory"
            };

            bool creatingSucceed = serviceEvaNN.CreateNetwork(trainConfig.MemoryFolder, netStructure);

            if (creatingSucceed)
            {
                serviceEvaNN.Train(trainConfig, 30, true);
                //serviceEvaNN.CalculateStatistic(trainConfig);
            }

            Console.WriteLine("Done");
            Console.ReadKey();
        }
    }
}
