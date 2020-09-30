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
                NeuronsByLayers = new[] { 2, 1 }
            };

            TrainingConfiguration trainConfig = new TrainingConfiguration
            {
                StartIteration = 0,
                EndIteration = 20000,
                InputDatasetFilename = "TrainingSets//inputSets.txt",
                OutputDatasetFilename = "TrainingSets//outputSets.txt",
                MemoryFolder = "Memory"
            };

            bool creatingSucceed = serviceEvaNN.CreateNetwork(trainConfig.MemoryFolder, netStructure);

            double[] netResult;

            if (creatingSucceed)
            {
                serviceEvaNN.Train(trainConfig, 20000, true);
                netResult = serviceEvaNN.Handle(new double[2] { 0, 0 });
            }

            Console.WriteLine("Done");
            Console.ReadKey();
        }
    }
}
