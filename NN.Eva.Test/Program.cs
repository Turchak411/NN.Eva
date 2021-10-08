using System;
using NN.Eva.Models;
using NN.Eva.Test.SimulationExamples;

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
            // Black Jack simulation example:
            SimulationBlackJack simulationBlackJack = new SimulationBlackJack();
            simulationBlackJack.StartTraining();
        }
    }
}
