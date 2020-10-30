using System;
using System.Collections.Generic;
using NN.Eva.Models;

namespace NN.Eva.Core.GeneticAlgorithm
{
    public class NeuralNetworkGeneticAlg
    {
        public List<LayerGeneticAlg> LayerList { get; set; } = new List<LayerGeneticAlg>();

        public NeuralNetworkGeneticAlg(List<double> weights, NetworkStructure networkStructure)
        {
            List<(int, int)> weightOnLayers = new List<(int, int)>();

            int index = 0;

            // Calculate weights count for the first layer:
            int countOfWeightsFirstLayer = networkStructure.NeuronsByLayers[0] * networkStructure.InputVectorLength;
            weightOnLayers.Add((index, countOfWeightsFirstLayer));
            index += countOfWeightsFirstLayer;

            // Calculate weights count for the other layers:
            for (int i = 1; i < networkStructure.NeuronsByLayers.Length; i++)
            {
                int countOfWeights = networkStructure.NeuronsByLayers[i] * networkStructure.NeuronsByLayers[i - 1];
                weightOnLayers.Add((index, countOfWeights));
                index += countOfWeights;
            }

            for (int i = 0; i < networkStructure.NeuronsByLayers.Length; i++)
            {
                LayerList.Add(new LayerGeneticAlg(weightOnLayers[i], weights, networkStructure.NeuronsByLayers[i]));
            }
        }

        /// <summary>
        /// Handling data
        /// </summary>
        /// <param name="data"></param>
        /// <param name="errorText"></param>
        /// <returns></returns>
        public double[] Handle(double[] data, ref string errorText)
        {
            // Check for non equaling of input length of data and network's receptors:
            if (data.Length != LayerList[0].GetWeights()[0].Length)
            {
                errorText = String.Format("Expected by network input-vector length: {0}\nInputed data-vector length: {1}", LayerList[0].GetWeights()[0].Length, data.Length);
                return null;
            }

            double[] tempData = data;

            for (int i = 0; i < LayerList.Count; i++)
            {
                tempData = LayerList[i].Handle(tempData);
            }

            // There is one double value at the last handle
            return tempData;
        }

        /// <summary>
        /// Handling data without vector's length check
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public double[] HandleUnsafe(double[] data)
        {
            double[] tempData = data;

            for (int i = 0; i < LayerList.Count; i++)
            {
                tempData = LayerList[i].Handle(tempData);
            }

            // There is one double value at the last handle
            return tempData;
        }
    }
}
