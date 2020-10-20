using System.Collections.Generic;
using NN.Eva.Models;
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

        /// <summary>
        /// Chromosome in NN context = Network's weights in row
        /// </summary>
        private List<List<double>> _chromosomeList = new List<List<double>>();

        private List<double> _funcValuesCollection = new List<double>();

        private List<List<double>> _newChromosomeList = new List<List<double>>();

        public void StartTraining(int iterationCount, bool unsafeMode)
        {
            int networkChromosomesCount = 20;

            GeneratePopulation(20);

            // TODO:

        }

        private void GeneratePopulation(int chromosomesCount)
        {
            ServiceWeightsGenerator serviceWeightsGenerator = new ServiceWeightsGenerator();

            for (int i = 0; i < chromosomesCount; i++)
            {
                _chromosomeList.Add(serviceWeightsGenerator.GenerateMemoryWeights(NetworkStructure));
            }
        }

        private void CreateNetworkByWeightsVector(List<double> networksWeightsVector)
        {
            FileManager fileManager = new FileManager(NetworkStructure);
            fileManager.SaveMemoryFromWeightsAndStructure(networksWeightsVector, NetworkStructure);
        }
    }
}
