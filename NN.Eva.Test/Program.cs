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
                InputVectorLength = 2,
                NeuronsByLayers = new[] { 1, 1 }
            };

            TrainingConfiguration trainConfig = new TrainingConfiguration
            {
                StartIteration = 100000,
                EndIteration = 200000,
                InputDatasetFilename = "TrainingSets//inputSets.txt",
                OutputDatasetFilename = "TrainingSets//outputSets.txt",
                MemoryFolder = "Memory"
            };

            bool creatingSucceed = serviceEvaNN.CreateNetwork(trainConfig.MemoryFolder, netStructure);

            if (creatingSucceed)
            {
                serviceEvaNN.Train(trainConfig, 100000, true);
            }

            Console.WriteLine("Done");
            Console.ReadKey();
        }
    }
}
