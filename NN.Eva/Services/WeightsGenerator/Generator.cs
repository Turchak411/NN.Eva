using System;
using System.IO;
using System.Collections.Generic;
using ConsoleProgressBar;
using NN.Eva.Models;

namespace NN.Eva.Services.WeightsGenerator
{
    public class Generator
    {
        public List<double> GenerateWeightsVector(NetworkStructure networkStructure, Random rnd)
        {
            List<double> weightsInVectors = new List<double>();

            // Generate input weights:
            for (int i = 0; i < networkStructure.NeuronsByLayers[0]; i++)
            {
                for (int k = 0; k < networkStructure.InputVectorLength; k++)
                {
                    weightsInVectors.Add(GenerateValue(rnd));
                }
            }

            // Generate other network neuron's weights:
            for (int i = 1; i < networkStructure.NeuronsByLayers.Length; i++)
            {
                for (int k = 0; k < networkStructure.NeuronsByLayers[i]; k++)
                {
                    for (int j = 0; j < networkStructure.NeuronsByLayers[i - 1]; j++)
                    {
                        weightsInVectors.Add(GenerateValue(rnd));
                    }
                }
            }

            return weightsInVectors;
        }

        public List<double> GenerateEmptyWeightsVector(NetworkStructure networkStructure)
        {
            List<double> weightsInVectors = new List<double>();

            // Generate input weights:
            for (int i = 0; i < networkStructure.NeuronsByLayers[0]; i++)
            {
                for (int k = 0; k < networkStructure.InputVectorLength; k++)
                {
                    weightsInVectors.Add(0);
                }
            }

            // Generate other network neuron's weights:
            for (int i = 1; i < networkStructure.NeuronsByLayers.Length; i++)
            {
                for (int k = 0; k < networkStructure.NeuronsByLayers[i]; k++)
                {
                    for (int j = 0; j < networkStructure.NeuronsByLayers[i - 1]; j++)
                    {
                        weightsInVectors.Add(0);
                    }
                }
            }

            return weightsInVectors;
        }

        public void GenerateMemory(int inputVectorLength, int[] netScheme, string filePath)
        {
            Random rnd = new Random(DateTime.Now.Millisecond);

            double offsetValue = 0.5;
            double offsetWeight = -1;

            using (var progress = new ProgressBar())
            {
                int iteration = 0;
                int iterationsTotal = CalcTotalIterations(netScheme);

                using (StreamWriter fileWriter = new StreamWriter(filePath))
                {
                    // Writing metadata of network:
                    fileWriter.Write(inputVectorLength);

                    for (int i = 0; i < netScheme.Length; i++)
                    {
                        fileWriter.Write(" " + netScheme[i]);
                    }

                    fileWriter.WriteLine();

                    // Writing memory:
                    for (int i = 0; i < netScheme.Length; i++)
                    {
                        for (int k = 0; k < netScheme[i]; k++)
                        {
                            fileWriter.Write("layer_{0} neuron_{1} {2} {3}", i, k, 
                                                                        offsetValue.ToString().Replace('.', ','),
                                                                        offsetWeight.ToString().Replace('.', ','));

                            if (i == 0)
                            {
                                GenerateValueRow(fileWriter, inputVectorLength, rnd);
                            }
                            else
                            {
                                GenerateValueRow(fileWriter, netScheme[i - 1], rnd);
                            }

                            iteration++;
                            progress.Report((double)iteration / iterationsTotal);
                        }
                    }
                }
            }
        }

        private int CalcTotalIterations(int[] netScheme)
        {
            int iterationsTotal = 0;

            foreach (var layer in netScheme)
            {
                iterationsTotal += layer;
            }

            return iterationsTotal;
        }

        private void GenerateValueRow(StreamWriter fileWriter, int valuesRowLength, Random rnd)
        {
            for (int i = 0; i < valuesRowLength; i++)
            {
                fileWriter.Write(" " + GenerateValue(rnd).ToString().Replace('.', ','));
            }

            fileWriter.WriteLine();
        }

        private double GenerateValue(Random rnd)
        {
            return rnd.NextDouble() - 0.5;
        }
    }
}
