using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using NN.Eva.Extensions;
using NN.Eva.Models;
using NN.Eva.Models.GeneticAlgorithm;
using NN.Eva.Services;
using NN.Eva.Services.WeightsGenerator;

namespace NN.Eva.Core.GeneticAlgorithm
{
    public class GeneticAlgorithmTeacher
    {
        public List<double[]> InputDatasets { get; set; }

        public List<double[]> OutputDatasets { get; set; }

        public NetworkStructure NetworkStructure { get; set; }

        #region Synchronization's objects

        private object _sync = new object();

        #endregion

        public void StartTraining(int iterationCount, bool unsafeMode, string memoryPath)
        {
            int networkChromosomesCount = 10;
            int newChromosomeCount = 5;
            int luckerCount = 3;

            Random rnd = new Random(DateTime.Now.Millisecond);

            List<FitnessFunction> fitnessFuncValues = new List<FitnessFunction>();

            // Generations list (actual and previously):
            // Chromosome in NN context = Network's weights in row
            List<List<double>> actualGeneration = GenerateExistentPopulation(networkChromosomesCount, memoryPath, newChromosomeCount);

            int currentIteration = 0;
            List<double> fitnessValuesTrace = new List<double>();

            do
            {
                // Selection:
                actualGeneration = DoSelection(actualGeneration);

                // Mutation:
                actualGeneration = DoMutation(actualGeneration);

                // Fitness function calculating:
                fitnessFuncValues = CalculateFitnessFunctionValues(actualGeneration, unsafeMode);
                // Sorting fitness values by value:
                fitnessFuncValues = fitnessFuncValues.OrderBy(x => x.Value).ToList();

                // Sorting generation by already sorted fitness values:
                List<List<double>> tempGenerationList = new List<List<double>>();

                // Adding with cutting generation by started generation count:
                for (int k = 0; k < networkChromosomesCount - luckerCount; k++)
                {
                    tempGenerationList.Add(actualGeneration[fitnessFuncValues[k].ChromosomeIndex]);
                }

                // Add "luckers":
                for (int i = 0; i < luckerCount; i++)
                {
                    int luckerIndex = rnd.Next(networkChromosomesCount - luckerCount - 1, fitnessFuncValues.Count);
                    tempGenerationList.Add(actualGeneration[fitnessFuncValues[luckerIndex].ChromosomeIndex]);
                }

                actualGeneration = tempGenerationList;

                double avgFitnessValue = fitnessFuncValues.Average(x => x.Value);

                // Add avg fitness value to trace:
                fitnessValuesTrace.Add(avgFitnessValue);
                //Remove trace tail if exist one:
                if (fitnessValuesTrace.Count > 100)
                {
                    fitnessValuesTrace.RemoveAt(0);
                }

                Console.WriteLine("Average generation {0} learning rate: {1}", currentIteration,
                                                                               avgFitnessValue);

                // Cataclysm:
                if (fitnessValuesTrace.Count == 100 && IsPopulationDegenerated(fitnessValuesTrace))
                {
                    Console.WriteLine("=========================\n" +
                                      "Corrective \"cataclysm\"!\n" +
                                      $"Removed: {actualGeneration.Count * 0.6}\n" +
                                      "=========================");

                    actualGeneration = DoCataclysm(actualGeneration, 0.6);

                    // Clear values trace for avoid multiple cataclysms by old data:
                    fitnessValuesTrace.Clear();
                }

                currentIteration++;
            } 
            while (currentIteration < iterationCount); // && fitnessFuncValues.Average(x => x.Value) > 0.00001);

            // Saving the best network:
            CreateNetworkMemoryFileByWeightsVector(actualGeneration[0]);
        }

        /// <summary>
        /// Generate new random population
        /// </summary>
        /// <param name="chromosomesCount"></param>
        /// <returns></returns>
        private List<List<double>> GeneratePopulation(int chromosomesCount)
        {
            List<List<double>> generation = new List<List<double>>();

            ServiceWeightsGenerator serviceWeightsGenerator = new ServiceWeightsGenerator();
            Random rnd = new Random(DateTime.Now.Millisecond);

            Parallel.For(0, chromosomesCount, i =>
            {
                lock (_sync)
                {
                    generation.Add(serviceWeightsGenerator.GenerateMemoryWeights(NetworkStructure, rnd));
                }
            });

            return generation;
        }

        /// <summary>
        /// Generate whole population with items from memory and random
        /// </summary>
        /// <param name="chromosomesCount"></param>
        /// <param name="memoryPath"></param>
        /// <param name="newChromosomesCount"></param>
        /// <returns></returns>
        private List<List<double>> GenerateExistentPopulation(int chromosomesCount, string memoryPath, int newChromosomesCount)
        {
            if(chromosomesCount < newChromosomesCount)
            {
                return new List<List<double>>();
            }

            List<List<double>> generation = new List<List<double>>();

            List<double> networkFromFile = FileManager.LoadWholeMemoryFile(memoryPath);

            for(int i = 0; i < chromosomesCount - newChromosomesCount; i++)
            {
                generation.Add(networkFromFile.CloneChromosome());
            }

            // Added random chromosomes:
            generation.AddRange(GeneratePopulation(newChromosomesCount));

            return generation;
        }

        /// <summary>
        /// Fitness function calculating
        /// </summary>
        /// <param name="generation"></param>
        /// <param name="unsafeMode"></param>
        /// <returns></returns>
        private List<FitnessFunction> CalculateFitnessFunctionValues(List<List<double>> generation, bool unsafeMode)
        {
            // Creating real neural networks by weights lists:
            List<NeuralNetworkGeneticAlg> networksList = generation.Select(t => new NeuralNetworkGeneticAlg(t, NetworkStructure)).ToList();

            // Calculating values:
            List<FitnessFunction> fitnessFuncValues = new List<FitnessFunction>();

            Parallel.For(0, networksList.Count, i =>
            {
                FitnessFunction fitnessFunction = new FitnessFunction
                {
                    ChromosomeIndex = i
                };

                fitnessFunction.CalculateValue(networksList[i], InputDatasets, OutputDatasets, unsafeMode);

                lock (_sync)
                {
                    fitnessFuncValues.Add(fitnessFunction);
                }
               
            });

            return fitnessFuncValues;
        }

        /// <summary>
        /// Selection operation
        /// </summary>
        /// <param name="generation"></param>
        /// <param name="selectionChance"></param>
        /// <returns></returns>
        private List<List<double>> DoSelection(List<List<double>> generation, double selectionChance = 0.64)
        {
            // Cloning generation to new generation list:
            List<List<double>> newGeneration = generation.CloneGeneration();

            Random rnd = new Random(DateTime.Now.Millisecond);

            for (int i = 0; i < generation.Count; i++)
            {
                // Choosing partner for selection:
                int randomPartnerNumber;

                do
                {
                    randomPartnerNumber = rnd.Next(generation.Count);
                }
                while (randomPartnerNumber == i);

                // Probe to selection by selection-chance:
                if (rnd.NextDouble() < selectionChance)
                {
                    // Then do selection >>> crossover operator:
                    newGeneration.Add(CrossoverOperator(generation[i], generation[randomPartnerNumber], rnd));
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
                if (rnd.NextDouble() <= localGenSelectionChance)
                {
                    // Adding gen from parent 0:
                    newChromosome.Add(chromosome0[i]);
                }
                else
                {
                    // Otherwise adding gen from parent 1:
                    newChromosome.Add(chromosome1[i]);
                }
            }

            return newChromosome;
        }

        /// <summary>
        /// Mutation operation
        /// </summary>
        /// <param name="generation"></param>
        /// <param name="mutationChance"></param>
        /// <returns></returns>
        private List<List<double>> DoMutation(List<List<double>> generation, double mutationChance = 0.01)
        {
            List<List<double>> newGeneration = generation.CloneGeneration();

            Random rnd = new Random(DateTime.Now.Millisecond);

            // Do mutation with mutation-chance (1%)
            newGeneration.AddRange(from t in generation 
                where rnd.NextDouble() < mutationChance
                select MutationOperator(t, rnd));

            return newGeneration;
        }

        /// <summary>
        /// Mutation operator
        /// </summary>
        /// <param name="chromosome"></param>
        /// <param name="rnd"></param>
        /// <returns></returns>
        private List<double> MutationOperator(List<double> chromosome, Random rnd)
        {
            return (from t in chromosome 
                let changingValue = (double) rnd.Next(-1000, 1000) / 100.0
                select t + changingValue).ToList();
        }

        /// <summary>
        /// Check for degenerated population
        /// </summary>
        /// <param name="fitnessValuesTrace"></param>
        /// <param name="valueRoundCount"></param>
        /// <param name="overwhelmingMajorityPercent"></param>
        /// <returns></returns>
        private bool IsPopulationDegenerated(List<double> fitnessValuesTrace, int valueRoundCount = 6, double overwhelmingMajorityPercent = 0.7)
        {
            return fitnessValuesTrace.GroupBy(x => Math.Round(x, valueRoundCount)).Any(g => g.Count() > fitnessValuesTrace.Count * overwhelmingMajorityPercent);
        }

        /// <summary>
        /// Cataclysm operation
        /// </summary>
        /// <param name="generation"></param>
        /// <param name="removingPercent"></param>
        /// <returns></returns>
        private List<List<double>> DoCataclysm(List<List<double>> generation, double removingPercent = 0.5)
        {
            // Thanos snap:
            int generationRemovedResultCount = (int)(generation.Count * removingPercent);

            ShuffleList(generation);

            // Creating queue by generation list:
            Queue<List<double>> generationInQueue = new Queue<List<double>>(generation);

            ServiceWeightsGenerator serviceWeightsGenerator = new ServiceWeightsGenerator();
            Random rnd = new Random(DateTime.Now.Millisecond);

            for (int i = 0; i < generationRemovedResultCount; i++)
            {
                generationInQueue.Dequeue();
                generationInQueue.Enqueue(serviceWeightsGenerator.GenerateMemoryWeights(NetworkStructure, rnd));
            }

            return generationInQueue.ToList();
        }

        /// <summary>
        /// Shuffling list
        /// </summary>
        /// <param name="generation"></param>
        private void ShuffleList(List<List<double>> generation)
        {
            Random rnd = new Random(DateTime.Now.Millisecond);

            for(int i = generation.Count; i > 0; i--)
            {
                // Swapping:
                generation.SwapItems(0, rnd.Next(0, i));
            }
        }

        /// <summary>
        /// Creating network memory by chromosome
        /// </summary>
        /// <param name="networksWeightsVector"></param>
        private void CreateNetworkMemoryFileByWeightsVector(List<double> networksWeightsVector)
        {
            FileManager.SaveMemoryFromWeightsAndStructure(networksWeightsVector, NetworkStructure);
        }
    }
}
