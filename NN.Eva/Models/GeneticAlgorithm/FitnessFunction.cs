using System.Collections.Generic;
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
            int testPassed = 0;
            int testFailed = 0;

            for (int i = 0; i < outputDatasets.Count; i++)
            {
                if (IsVectorsRoughlyEquals(outputDatasets[i], network.Handle(inputDatasets[i]), 0.15))
                {
                    testPassed++;
                }
                else
                {
                    testFailed++;
                }
            }

            Value = (double)testPassed * 100 / (testPassed + testFailed);
        }

        private bool IsVectorsRoughlyEquals(double[] sourceVector0, double[] controlVector1, double equalsPercent)
        {
            // Возвращение неравенства, если длины векторов не совпадают
            if (sourceVector0.Length != controlVector1.Length)
            {
                return false;
            }

            for (int i = 0; i < sourceVector0.Length; i++)
            {
                if (controlVector1[i] < sourceVector0[i] - equalsPercent || controlVector1[i] > sourceVector0[i] + equalsPercent)
                {
                    return false;
                }
            }

            return true;
        }
    }
}
