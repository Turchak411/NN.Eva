using NN.Eva.Models;
using NN.Eva.Services;
using System;
using System.Collections.Generic;

namespace NN.Eva.Core
{
    public class SingleNetworkTeacher
    {
        public int Id { get; set; }

        public NeuralNetwork Network { get; set; }

        public TrainingConfiguration TrainingConfiguration { get; set; }

        public List<double[]> InputDatasets { get; set; }

        public List<double[]> OutputDatasets { get; set; }

        public Logger Logger { get; set; }

        private int Iteration = 0;

        public void Train()
        {
            if (Network == null) return;
            if (TrainingConfiguration == null) return;

            Iteration = TrainingConfiguration.EndIteration;
            if (Iteration == 0 || TrainingConfiguration.EndIteration - TrainingConfiguration.StartIteration <= 0) return;

            if (InputDatasets == null) return;
            if (OutputDatasets == null) return;

            for (int iteration = TrainingConfiguration.StartIteration; iteration < Iteration; iteration++)
            {
                // Calculating learn-speed rate:
                var learningSpeed = 0.01 * Math.Pow(0.1, iteration / 150000);
                for (int k = 0; k < InputDatasets.Count; k++)
                {
                    Network.Handle(InputDatasets[k]);

                    // Передает для обучения только 1 элемент выходного вектора
                    // (Класс на который конкретной сети нужно активироваться)
                    double[] outputDataSetArray = { OutputDatasets[k][Id] };

                    try
                    {
                        Network.Teach(InputDatasets[k], outputDataSetArray, learningSpeed);
                    }
                    catch (Exception ex)
                    {
                        Logger.LogError(ErrorType.TrainError, ex);
                        return;
                    }
                }
            }

            Network.SaveMemory(TrainingConfiguration.MemoryFolder + "//memory_" + Id + ".txt");
        }
    }
}
