using System;
using NN.Eva.RL;
using NN.Eva.Models;
using NN.Eva.Models.RL;
using System.Collections.Generic;

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

            RLConfigModel configModel = new RLConfigModel
            {
                ActionsCount = 2,
                PositivePrice = 0.1,
                NegativePrice = -0.1,
                MainTailMaxLength = 2,
                FantomTailMaxLength = 2
            };

            // Agent = Deep Neural Network
            int dataVectorLength = 1;

            NetworkStructure netStructure = new NetworkStructure(configModel, dataVectorLength)
            {
                NeuronsByLayers = new[] { 3, 3, 1 }
            };

            TrainingConfigurationLite trainConfig = new TrainingConfigurationLite
            {
                EndIteration = 100000,
                MemoryFolder = "Memory"
            };

            bool creatingSucceed = serviceEvaRL.CreateAgent(trainConfig, configModel, netStructure);

            if (creatingSucceed)
            {
                EnvironmentInteractionProcessStart(serviceEvaRL);
            }
        }

        private static void EnvironmentInteractionProcessStart(ServiceEvaRL serviceEvaRL)
        {
            // Black Jack example:
            // In this example failure environment will be single value-sum, not vector of parameters, when Agent "dies"/fails
            // Agent's decisions will be: Grab another one & Stop

            RLWorkingModel workingModel = new RLWorkingModel
            {
                CurrentEnvironment = new double[1] { 0 }, // Values sum
                FailureEnvironment = new double[1] { 21 } // End game sum
            };

            List<int> cardValuesPull = new List<int>()
            {
                2, 2, 2, 2,
                3, 3, 3, 3,
                4, 4, 4, 4,
                5, 5, 5, 5,
                6, 6, 6, 6,
                7, 7, 7, 7,
                8, 8, 8, 8,
                9, 9, 9, 9,
                10, 10, 10, 10,
                2, 2, 2, 2,
                3, 3, 3, 3,
                4, 4, 4, 4,
                11, 11, 11, 11
            };

            Random rnd = new Random(DateTime.Now.Millisecond);

            while(true)
            {
                double[] agentDecision = serviceEvaRL.TrainingTick(workingModel);

                // If Agent currently retrained:
                if(agentDecision == null)
                {
                    // Reset environment for new game:
                    workingModel.CurrentEnvironment = new double[1] { 0 };

                    // Re-ask Agent:
                    agentDecision = serviceEvaRL.TrainingTick(workingModel);
                }

                // 0 - Grab another one
                // 1 - Stop
                if(agentDecision[0] >= agentDecision[1])
                {
                    // Grab another one

                    // Pull out another card:
                    int cardValue = cardValuesPull[rnd.Next(cardValuesPull.Count)];

                    Console.WriteLine("Agent wants to grab another one card... Card value is " + cardValue + ".");

                    // Updating environment:
                    workingModel.CurrentEnvironment[0] += cardValue;

                    Console.WriteLine("Current sum is: {0:f0}", workingModel.CurrentEnvironment[0]);

                    // Check for game ending
                    if (workingModel.CurrentEnvironment[0] == workingModel.FailureEnvironment[0])
                    {
                        Console.WriteLine("Agent won!\nGame end.");
                        break;
                    }

                    // Checking for value overflowing:
                    if (workingModel.CurrentEnvironment[0] > workingModel.FailureEnvironment[0])
                    {
                        workingModel.CurrentEnvironment[0] = workingModel.FailureEnvironment[0];
                        Console.WriteLine("Value overflowed!\nGame end.");
                    }
                }
                else
                {
                    // Stop
                    Console.WriteLine("Agent wants to stop.\nGame end.");
                    break;
                }
            }


        }
    }
}
