using System.Collections.Generic;
using System.Linq;
using NN.Eva.Core.GeneticAlgorithm;

namespace NN.Eva.Models.GeneticAlgorithm
{
    public class FitnessFunction
    {
        /// <summary>
        /// Function value - network's training accuracy
        /// </summary>
        public double Value { get; private set; }

        /// <summary>
        /// Chromosome index in generation
        /// </summary>
        public int ChromosomeIndex { get; set; }

        public void CalculateValue(HandleOnlyNN network, List<double[]> inputDatasets, List<double[]> outputDatasets)
        {
            var netAnswers = inputDatasets.Select(network.Handle).ToList();
            Value = RecalculateEpochError(netAnswers, outputDatasets);
        }

        private double RecalculateEpochError(List<double[]> netResultList, List<double[]> outputDatasets)
        {
            double sum = 0;

            for (int i = 0; i < netResultList.Count; i++)
            {
                for (int k = 0; k < netResultList[i].Length; k++)
                {
                    double delta = outputDatasets[i][k] - netResultList[i][k];
                    sum += delta * delta;
                }
            }

            return sum / outputDatasets.Count;
        }
    }
}
