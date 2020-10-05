using System;
using System.Diagnostics;
using NN.Eva.Models;

namespace NN.Eva.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            ServiceEvaNN serviceEvaNN = new ServiceEvaNN();

            NetworkStructure netStructure = new NetworkStructure
            {
                InputVectorLength = 10,
                NeuronsByLayers = new[] { 230, 150, 120, 1 }
            };

            TrainingConfiguration trainConfig = new TrainingConfiguration
            {
                StartIteration = 0,
                EndIteration = 5,
                InputDatasetFilename = "TrainingSets//inputSets.txt",
                OutputDatasetFilename = "TrainingSets//outputSets.txt",
                MemoryFolder = "Memory"
            };

            bool creatingSucceed = serviceEvaNN.CreateNetwork(trainConfig.MemoryFolder, netStructure);

            if (creatingSucceed)
            {
                serviceEvaNN.Train(trainConfig,
                                  5,
                                 false,
                                   ProcessPriorityClass.Normal,
                                  true);
                //serviceEvaNN.CalculateStatistic(trainConfig);
            }

            Console.WriteLine("Done!");

            stopWatch.Stop();
            TimeSpan ts = stopWatch.Elapsed;

            string elapsedTime = String.Format("{0:00}:{1:00}", ts.Minutes, ts.Seconds);
            Console.WriteLine("Time spend: " + elapsedTime);

            Console.ReadKey();
        }
    }
}
