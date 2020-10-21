using System;
using System.Collections.Generic;
using System.Linq;
using NN.Eva.Models;
using NN.Eva.Models.GeneticAlgorithm;
using NN.Eva.Services;
using NN.Eva.Services.WeightsGenerator;

namespace NN.Eva.Core.GeneticAlgorithm
{
    public class GeneticAlgorithmTeacher
    {
        public NeuralNetwork Network { get; private set; } = null;

        public List<double[]> InputDatasets { get; set; }

        public List<double[]> OutputDatasets { get; set; }

        public NetworkStructure NetworkStructure { get; set; }

        public void StartTraining(int iterationCount)
        {
            int networkChromosomesCount = 20;

            List<FitnessFunction> fitnessFuncValues = new List<FitnessFunction>();

            // Generations list (actual and previously):
            // Chromosome in NN context = Network's weights in row
            List<List<double>> actualGeneration = GeneratePopulation(networkChromosomesCount);

            for (int i = 0; i < iterationCount; i++)
            {
                // Selection:
                actualGeneration = DoSelection(actualGeneration);

                // Mutation:
                actualGeneration = DoMutation(actualGeneration);

                // Fitness function calculating:
                fitnessFuncValues = CalculateFitnessFunctionValues(actualGeneration);
                // Sorting fitness values by value:
                fitnessFuncValues = fitnessFuncValues.OrderBy(x => x.Value).ToList();

                 // Sorting generation by already sorted fitness values:
                List<List<double>> tempGenerationList = new List<List<double>>();

                for (int k = 0; k < fitnessFuncValues.Count; k++)
                {
                    tempGenerationList.Add(actualGeneration[fitnessFuncValues[k].ChromosomeIndex]);
                }

                actualGeneration = tempGenerationList;

                // Cutting generation to initial count:
                try
                {
                    actualGeneration.RemoveRange(networkChromosomesCount - 1,
                                                 actualGeneration.Count - networkChromosomesCount);
                }
                catch { }
            }
        }

        private List<List<double>> GeneratePopulation(int chromosomesCount)
        {
            List<List<double>> generation = new List<List<double>>();

            ServiceWeightsGenerator serviceWeightsGenerator = new ServiceWeightsGenerator();

            for (int i = 0; i < chromosomesCount; i++)
            {
                generation.Add(serviceWeightsGenerator.GenerateMemoryWeights(NetworkStructure));
            }

            return generation;
        }

        private List<FitnessFunction> CalculateFitnessFunctionValues(List<List<double>> generation)
        {
            // Creating real neural networks by weights lists:
            List<HandleOnlyNN> networksList = new List<HandleOnlyNN>(); 
            
            for (int i = 0; i < generation.Count; i++)
            {
                networksList.Add(new HandleOnlyNN(generation[i]));
            }

            // Calculating values:
            List<FitnessFunction> fitnessFuncValues = new List<FitnessFunction>();

            for (int i = 0; i < networksList.Count; i++)
            {
                FitnessFunction fitnessFunction = new FitnessFunction
                {
                    ChromosomeIndex = i
                };

                fitnessFunction.CalculateValue(networksList[i], InputDatasets, OutputDatasets);

                fitnessFuncValues.Add(fitnessFunction);
            }

            return fitnessFuncValues;
        }

        private List<List<double>> DoSelection(List<List<double>> generation, double selectionChance = 0.64)
        {
            List<List<double>> newGeneration = new List<List<double>>();

            Random rnd = new Random(DateTime.Now.Millisecond);

            for (int i = 0; i < generation.Count; i++)
            {
                for (int k = 0; k < generation.Count; k++)
                {
                    if(rnd.NextDouble() < selectionChance &&
                       i != k)
                    {
                        // Then do selection >>> crossover operator:
                        newGeneration.Add(CrossoverOperator(generation[i], generation[k], rnd));
                    }
                }
            }

            return newGeneration;
        }

        /// <summary>
        /// Crossover operator
        /// P.S. processing different in genetic alg + neural networks
        /// </summary>
        /// <param name="chromosome0"></param>
        /// <param name="chromosome1"></param>
        /// <param name="rnd"></param>
        /// <returns></returns>
        private List<double> CrossoverOperator(List<double> chromosome0, List<double> chromosome1, Random rnd)
        {
            double localGenSelectionChance = 0.5;

            List<double> newChromosome = new List<double>();

            for (int i = 0; i < chromosome0.Count; i++)
            {
                if (rnd.NextDouble() < localGenSelectionChance)
                {
                    // Adding gen from parent 0:
                    newChromosome.Add(chromosome1[i]);
                }
                else
                {
                    // Otherwise adding gen from parent 1:
                    newChromosome.Add(chromosome1[i]);
                }
            }

            return newChromosome;
        }

        private List<List<double>> DoMutation(List<List<double>> generation, double mutationChance = 0.01)
        {
            List<List<double>> newGeneration = new List<List<double>>();

            Random rnd = new Random(DateTime.Now.Millisecond);

            for (int i = 0; i < generation.Count; i++)
            {
                if (rnd.NextDouble() < mutationChance)
                {
                    // Then do selection >>> crossover operator:
                    newGeneration.Add(MutationOperator(generation[i], rnd));
                }
            }

            return newGeneration;
        }

        private List<double> MutationOperator(List<double> chromosome, Random rnd)
        {
            List<double> newChromosome = new List<double>();

            for (int i = 0; i < chromosome.Count; i++)
            {
                double changingValue = (double)rnd.Next(-100, 100) / 100.0;

                newChromosome[i] = chromosome[i] + changingValue;
            }

            return newChromosome;
        }

        private void CreateNetworkByWeightsVector(List<double> networksWeightsVector)
        {
            FileManager fileManager = new FileManager(NetworkStructure);
            fileManager.SaveMemoryFromWeightsAndStructure(networksWeightsVector, NetworkStructure);
        }
    }
}
