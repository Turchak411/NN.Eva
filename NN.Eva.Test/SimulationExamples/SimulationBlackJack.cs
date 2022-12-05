using NN.Eva.Models;
using NN.Eva.Models.RL;
using NN.Eva.RL;
using System;
using System.Collections.Generic;

namespace NN.Eva.Test.SimulationExamples
{
    public class SimulationBlackJack
    {
        public void Start()
        {
            ServiceEvaRL serviceEvaRL = new ServiceEvaRL();

            RLConfigModel configModel = new RLConfigModel
            {
                ActionsCount = 2
            };

            // Agent = Deep Neural Network
            int inputDataVectorLength = 1;

            NetworkStructure netStructure = new NetworkStructure(configModel, inputDataVectorLength)
            {
                NeuronsByLayers = new[] { 1, 1 } // 1 neurons number at the output required
            };

            string memoryFolder = "Memory";

            bool creatingSucceed = serviceEvaRL.CreateAgent(configModel, netStructure, memoryFolder);

            if (creatingSucceed)
            {
                EnvironmentInteractionProcessStart(serviceEvaRL);
            }
        }

        private void EnvironmentInteractionProcessStart(ServiceEvaRL serviceEvaRL)
        {
            // Black Jack example:
            // In this example failure environment will be single value-sum, not vector of parameters, when Agent "dies"/fails
            // Agent's decisions will be: Grab another one & Stop

            RLWorkingModel workingModel = new RLWorkingModel
            {
                CurrentEnvironment = new double[1] { 0 },   // Values sum
                FailureEnvironment = new double[1] { 0.21 } // End game sum
            };

            List<double> cardValuesPull = new List<double>()
            {
                0.02, 0.02, 0.02, 0.02,
                0.03, 0.03, 0.03, 0.03,
                0.04, 0.04, 0.04, 0.04,
                0.05, 0.05, 0.05, 0.05,
                0.06, 0.06, 0.06, 0.06,
                0.07, 0.07, 0.07, 0.07,
                0.08, 0.08, 0.08, 0.08,
                0.09, 0.09, 0.09, 0.09,
                0.1, 0.1, 0.1, 0.1,
                0.02, 0.02, 0.02, 0.02,
                0.03, 0.03, 0.03, 0.03,
                0.04, 0.04, 0.04, 0.04,
                0.11, 0.11, 0.11, 0.11
            };

            Random rnd = new Random(DateTime.Now.Millisecond);

            while (true)
            {
                double[] agentDecision;

                try
                {
                    agentDecision = serviceEvaRL.UseAgent(workingModel);
                }
                catch
                {
                    break;
                }

                // 0 - Grab another one
                // 1 - Stop
                if (agentDecision[0] >= agentDecision[1])
                {
                    // Grab another one

                    // Pull out another card:
                    double cardValue = cardValuesPull[rnd.Next(cardValuesPull.Count)];

                    Console.WriteLine("Agent wants to grab another one card... Card value is {0:f0}.", cardValue * 100);

                    // Updating environment:
                    workingModel.CurrentEnvironment[0] += cardValue;

                    Console.WriteLine("Current sum is: {0:f0}", workingModel.CurrentEnvironment[0] * 100);

                    // Check for game ending
                    if ((int)(workingModel.CurrentEnvironment[0] * 100) == (int)(workingModel.FailureEnvironment[0] * 100))
                    {
                        Console.WriteLine("Agent won!\nGame end.");
                        break;
                    }

                    // Checking for value overflowing:
                    if (workingModel.CurrentEnvironment[0] > workingModel.FailureEnvironment[0])
                    {
                        Console.WriteLine("Value overflowed!\nGame end.");
                        break;
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

        public void StartTraining()
        {
            ServiceEvaRL serviceEvaRL = new ServiceEvaRL();

            RLConfigModelTraining configModel = new RLConfigModelTraining
            {
                GreedyChance = 0.1,
                ActionsCount = 2,
                PositivePrice = 1,
                NegativePrice = -1,
                MainTailMaxLength = 1,
                FantomTailMaxLength = 3
            };

            // Agent = Deep Neural Network
            int inputDataVectorLength = 1;

            NetworkStructure netStructure = new NetworkStructure(configModel, inputDataVectorLength)
            {
                NeuronsByLayers = new[] { 1, 1 }
            };

            TrainingConfigurationLite trainConfig = new TrainingConfigurationLite
            {
                EndIteration = 100,
                MemoryFolder = "Memory"
            };

            bool creatingSucceed = serviceEvaRL.CreateAgent(configModel, netStructure, trainConfig);

            if (creatingSucceed)
            {
                EnvironmentInteractionTrainingProcessStart(serviceEvaRL);
            }
        }

        private void EnvironmentInteractionTrainingProcessStart(ServiceEvaRL serviceEvaRL)
        {
            // Black Jack example:
            // In this example failure environment will be single value-sum, not vector of parameters, when Agent "dies"/fails
            // Agent's decisions will be: Grab another one & Stop

            RLWorkingModel workingModel = new RLWorkingModel
            {
                CurrentEnvironment = new double[1] { 0 },   // Values sum
                FailureEnvironment = new double[1] { 0.21 } // End game sum
            };

            List<double> cardValuesPull = new List<double>()
            {
                0.02, 0.02, 0.02, 0.02,
                0.03, 0.03, 0.03, 0.03,
                0.04, 0.04, 0.04, 0.04,
                0.05, 0.05, 0.05, 0.05,
                0.06, 0.06, 0.06, 0.06,
                0.07, 0.07, 0.07, 0.07,
                0.08, 0.08, 0.08, 0.08,
                0.09, 0.09, 0.09, 0.09,
                0.1, 0.1, 0.1, 0.1,
                0.02, 0.02, 0.02, 0.02,
                0.03, 0.03, 0.03, 0.03,
                0.04, 0.04, 0.04, 0.04,
                0.11, 0.11, 0.11, 0.11
            };

            Random rnd = new Random(DateTime.Now.Millisecond);

            int winsCount = 0;
            int totalGamesCount = 0;

            while (true)
            {
                double[] agentDecision;

                try
                {
                    agentDecision = serviceEvaRL.TrainingTick(workingModel);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex);
                    break;
                }

                // If Agent currently retrained:
                if (agentDecision == null)
                {
                    // Reset environment for new game:
                    workingModel.CurrentEnvironment = new double[1] { 0 };

                    // Re-ask Agent:
                    agentDecision = serviceEvaRL.TrainingTick(workingModel);
                }

                // 0 - Grab another one
                // 1 - Stop
                if (agentDecision[0] >= agentDecision[1])
                {
                    // Grab another one

                    // Pull out another card:
                    double cardValue = cardValuesPull[rnd.Next(cardValuesPull.Count)];

                    Console.WriteLine("Agent wants to grab another one card... Card value is {0:f0}.", cardValue * 100);

                    // Updating environment:
                    workingModel.CurrentEnvironment[0] += cardValue;

                    Console.WriteLine("Current sum is: {0:f0}", workingModel.CurrentEnvironment[0] * 100);

                    // Check for game ending
                    if ((int)(workingModel.CurrentEnvironment[0] * 100) == (int)(workingModel.FailureEnvironment[0] * 100))
                    {
                        winsCount++;
                        Console.WriteLine("Agent won!\nGame end.");
                        PrintWinrate(winsCount, totalGamesCount);
                    }

                    // Checking for value overflowing:
                    if (workingModel.CurrentEnvironment[0] > workingModel.FailureEnvironment[0])
                    {
                        workingModel.CurrentEnvironment[0] = workingModel.FailureEnvironment[0];
                        totalGamesCount++;
                        Console.WriteLine("Value overflowed!\nGame end.");
                        PrintWinrate(winsCount, totalGamesCount);
                    }
                }
                else
                {
                    // Stop
                    totalGamesCount++;

                    if (workingModel.CurrentEnvironment[0] > 0.15)
                    {
                        winsCount++;
                    }

                    workingModel.CurrentEnvironment[0] = workingModel.FailureEnvironment[0];

                    Console.WriteLine("Agent wants to stop.\nGame end.");
                    PrintWinrate(winsCount, totalGamesCount);
                }
            }
        }

        private static void PrintWinrate(int winsCount, int totalGamesCount)
        {
            Console.WriteLine("Current winrate: {0:f0}", (double)winsCount / totalGamesCount * 100);
        }
    }
}