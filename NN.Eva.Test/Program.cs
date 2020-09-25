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
                NeuronsByLayers = new[] { 120, 110, 1 }
            };

            TrainingConfiguration trainConfig = new TrainingConfiguration
            {
                StartIteration = 0,
                EndIteration = 10,
                InputDatasetFilename = "TrainingSets//inputSets.txt",
                OutputDatasetFilename = "TrainingSets//outputSets.txt",
                MemoryFolder = "Memory"
            };

            serviceEvaNN.CreateNetwork(trainConfig.MemoryFolder, netStructure);

            serviceEvaNN.Train(trainConfig, 10);

            Console.WriteLine("Done");
            Console.ReadKey();
        }
    }
}
