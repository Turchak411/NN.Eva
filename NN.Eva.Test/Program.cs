using System;
using NN.Eva.Models;
using NN.Eva.RL;

namespace NN.Eva.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            // Network testing:
            //TestServiceNN();

            // RL-Agent testing:
            TestServiceRL();

            Console.WriteLine("Done!");
            Console.ReadKey();
        }

        private static void TestServiceNN()
        {
            ServiceEvaNN serviceEvaNN = new ServiceEvaNN();

            NetworkStructure netStructure = new NetworkStructure
            {
                InputVectorLength = 10,
                NeuronsByLayers = new[] { 230, 180, 160, 80, 1 },
                Alpha = 5
            };

            TrainingConfiguration trainingConfig = new TrainingConfiguration
            {
                TrainingAlgorithmType = TrainingAlgorithmType.RProp,
                StartIteration = 0,
                EndIteration = 10,
                InputDatasetFilename = "TrainingSets//inputSets.txt",
                OutputDatasetFilename = "TrainingSets//outputSets.txt",
                MemoryFolder = "Memory"
            };

            bool creatingSucceed = serviceEvaNN.CreateNetwork(trainingConfig.MemoryFolder, netStructure);

            if (creatingSucceed)
            {
                serviceEvaNN.CalculateStatistic(trainingConfig);
                //serviceEvaNN.Train(trainingConfig,
                //                   true,
                //                   ProcessPriorityClass.Normal,
                //                   true);
                //serviceEvaNN.CalculateStatistic(trainingConfig);
            }
        }

        private static void TestServiceRL()
        {
            ServiceEvaRL serviceEvaRL = new ServiceEvaRL();

            // Agent = Deep Neural Network
            NetworkStructure netStructure = new NetworkStructure
            {
                InputVectorLength = 10,
                NeuronsByLayers = new[] { 230, 180, 160, 80, 1 },
                Alpha = 5
            };

            TrainingConfiguration trainConfig = new TrainingConfiguration
            {
                StartIteration = 0,
                EndIteration = 100,
                InputDatasetFilename = "TrainingSets//inputSets.txt",
                OutputDatasetFilename = "TrainingSets//outputSets.txt",
                MemoryFolder = "Memory"
            };

            bool creatingSucceed = serviceEvaRL.CreateAgent(trainConfig, netStructure);

            if (creatingSucceed)
            {
                EnvironmentInteractionProcessStart();
            }
        }

        private static void EnvironmentInteractionProcessStart()
        {
            // *EnvironmentInteractionProcess here*
        }
    }
}
