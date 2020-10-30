using System.Collections.Generic;
using System.Linq;
using NN.Eva.Models;

namespace NN.Eva.Core.GeneticAlgorithm
{
    public class HandleOnlyLayer
    {
        public List<HandleOnlyNeuron> NeuronList { get; set; } = new List<HandleOnlyNeuron>();

        public HandleOnlyLayer((int, int) weightsOnLayer, List<double> weights, int neuronsCount)
        {
            int indexIntWeightList = weightsOnLayer.Item1;

            for (int i = 0; i < neuronsCount; i++)
            {
                int countOfNeuronWeight = weightsOnLayer.Item2 / neuronsCount;

                HandleOnlyNeuron neuron = new HandleOnlyNeuron
                {
                    OffsetValue = 0.5,
                    OffsetWeight = -1,
                    ActivationFunctionType = ActivationFunction.Sigmoid,
                    Weights = weights.GetRange(indexIntWeightList, countOfNeuronWeight).ToArray()
                };

                indexIntWeightList += countOfNeuronWeight;

                NeuronList.Add(neuron);
            }
        }

        public double[] Handle(double[] data)
        {
            double[] layerResultVector = new double[NeuronList.Count];

            for (int i = 0; i < layerResultVector.Length; i++)
            {
                layerResultVector[i] = NeuronList[i].Handle(data);
            }

            return layerResultVector;
        }

        public double[][] GetWeights()
        {
            double[][] weights = new double[NeuronList.Count][];

            for (int i = 0; i < NeuronList.Count; i++)
            {
                weights[i] = NeuronList[i].Weights;
            }

            return weights;
        }
    }
}
