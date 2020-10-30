using System;
using NN.Eva.Core.ResilientPropagation;
using NN.Eva.Core.ResilientPropagation.ActivationFunctions;
using NN.Eva.Models;
using NN.Eva.Services;

namespace NN.Eva.Test.RProp
{
    class Program
    {
        static void Main(string[] args)
        {
            var netStructure = new NetworkStructure
            {
                InputVectorLength = 10,
                NeuronsByLayers = new[] { 230, 150, 120, 1 }
            };

            var inputDataSets = FileManager.LoadTrainingDataset("TrainingSets//inputSets.txt").ToArray();
            var outputDataSets = FileManager.LoadTrainingDataset("TrainingSets//outputSets.txt").ToArray();

            var neuralNetworkRProp = new NeuralNetworkRProp(new ActivationSigmoid(), netStructure);
            
            double count = 0;
            double error = 0.0;
            double iteration = 0.0;
            while (iteration < 1000)
            {
                error = neuralNetworkRProp.Train(inputDataSets, outputDataSets);
                if (count > 500)
                {
                    count = 0;
                    Console.WriteLine($"Iteration: {iteration}\t Error: {error}");
                }
                Console.WriteLine($"Iteration: {iteration}\t Error: {error}");
                count++;
                iteration++;
            }

            Console.ReadKey();
        }
    }
}
