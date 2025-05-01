using System;
using System.Collections.Generic;
using System.Diagnostics;
using NN.Eva.Models;

namespace NN.Eva.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            TestMainNN();

            Console.WriteLine("Done!");
            Console.ReadKey();
        }

        private static void TestMainNN()
        {
            ServiceEvaNN serviceEvaNN = new ServiceEvaNN();

            NetworkStructure netStructure = new NetworkStructure
            {
                InputVectorLength = 10,
                NeuronsByLayers = new[] { 40, 80, 1 },
                Alpha = 1
            };

            TrainingConfiguration trainConfig = new TrainingConfiguration
            {
                TrainingAlgorithmType = TrainingAlgorithmType.BProp,
                StartIteration = 400_000,
                EndIteration = 500_000,
                InputDatasetFilename = "TrainingSets//inputSets.txt",
                OutputDatasetFilename = "TrainingSets//outputSets.txt",
                MemoryFolder = "Memory",
                ValidationSetSize = 10
            };

            bool creatingSucceed = serviceEvaNN.CreateNetwork(trainConfig.MemoryFolder, netStructure);

            if (creatingSucceed)
            {
                serviceEvaNN.Train(trainConfig,
                                   true,
                                   ProcessPriorityClass.Normal,
                                   true);
                serviceEvaNN.CheckDatasetsVectorsSimilarity(trainConfig);
            }

            var days = 1;

            var listValues = new List<double>() { 0.21940, 0.22339, 0.22535, 0.22557, 0.22499, 0.22508, 0.21980, 0.22559, 0.24359, 0.23859 };

            for (int i = 0; i < days; i++)
            {
                var resultValue = serviceEvaNN.Handle(listValues.ToArray());

                Console.WriteLine(resultValue[0] * 1000);

                listValues.RemoveAt(0);
                listValues.Add(resultValue[0]);
            }

            Console.WriteLine("Done!");
            Console.ReadKey();
        }
    }
}
