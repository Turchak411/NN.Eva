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
                InputVectorLength = 15,
                NeuronsByLayers = new[] { 235, 120, 110, 1 }
            };

            TrainingConfiguration trainConfig = new TrainingConfiguration
            {
                StartIteration = 0,
                EndIteration = 1000,
                InputDatasetFilename = "TrainingSets//inputSets.txt",
                OutputDatasetFilename = "TrainingSets//outputSets.txt",
                MemoryFolder = "Memory"
            };

            serviceEvaNN.CreateNetwork(trainConfig.MemoryFolder, netStructure, 2);

            //serviceEvaNN.Train(trainConfig, 1000);

            Console.WriteLine("Done");
            Console.ReadKey();
        }
    }
}
