using System;
using System.Threading;
using System.Threading.Tasks;
using NN.Eva.Core.ResilientPropagation.ActivationFunctions;
using NN.Eva.Core.ResilientPropagation.Layers;
using NN.Eva.Models;

namespace NN.Eva.Core.ResilientPropagation
{
    public class MultiThreadNeuralNetworkRProp : NeuralNetworkRProp
    {

        private object _lockNetwork = new object();
        private ThreadLocal<double[][]> _networkErrors;
        private ThreadLocal<double[][]> _networkOutputs;

        public MultiThreadNeuralNetworkRProp(IActivationFunction function, NetworkStructure networkStructure) : base(
            function, networkStructure)
        {
            _networkOutputs = new ThreadLocal<double[][]>(() => new double[Layers.Length][]);

            _networkErrors = new ThreadLocal<double[][]>(() =>
            {
                var e = new double[Layers.Length][];
                for (int i = 0; i < e.Length; i++)
                    e[i] = new double[Layers[i].Neurons.Length];
                return e;
            });
        }


        /// <summary>
        /// Train network
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <returns></returns>
        public override double Train(double[][] input, double[][] output)
        {
            ResetGradient();
            var lockSum = new object();
            double sumOfSquaredErrors = 0;

            Parallel.For(0, input.Length, () => 0.0,
                (i, loopState, partialSum) =>
                {
                    TeacherNetwork(input, i);
                    partialSum += CalculateError(output[i]);
                    CalculateGradient(input[i]);
                    return partialSum;
                },
                // Reduce
                (partialSum) =>
                {
                    lock (lockSum) sumOfSquaredErrors += partialSum;
                }
            );

            UpdateNetwork();
            return sumOfSquaredErrors/input.Length;
        }

        private void TeacherNetwork(double[][] input, int index)
        {
            double[] output = input[index];
            // compute each layer
            for (int i = 0; i < Layers.Length; i++)
            {
                output = Layers[i].Compute(output);
            }

            lock (_lockNetwork)
            {
                // Copy network outputs to local thread
                var networkOutputs = _networkOutputs.Value;
                for (int j = 0; j < networkOutputs.Length; j++)
                    networkOutputs[j] = Layers[j].Output;
            }
        }


        public void Reset(double rate)
        {
            Parallel.For(0, WeightsUpdates.Length, i =>
            {
                for (int j = 0; j < WeightsUpdates[i].Length; j++)
                for (int k = 0; k < WeightsUpdates[i][j].Length; k++)
                    WeightsUpdates[i][j][k] = rate;

                for (int j = 0; j < ThresholdsUpdates[i].Length; j++)
                    ThresholdsUpdates[i][j] = rate;
            });
        }


        /// <summary>
        /// Resets the gradient vector back to zero.
        /// </summary>
        protected override void ResetGradient()
        {
            Parallel.For(0, WeightsDerivatives.Length, i =>
            {
                for (int j = 0; j < WeightsDerivatives[i].Length; j++)
                    Array.Clear(WeightsDerivatives[i][j], 0, WeightsDerivatives[i][j].Length);
                Array.Clear(ThresholdsDerivatives[i], 0, ThresholdsDerivatives[i].Length);
            });
        }


        protected override double CalculateError(double[] desiredOutput)
        {
            double sumOfSquaredErrors = 0.0;
            int layersCount = Layers.Length;

            double[][] networkErrors = _networkErrors.Value;
            double[][] networkOutputs = _networkOutputs.Value;

            // Assume that all network neurons have the same activation function
            IActivationFunction function = Layers[0].Neurons[0].ActivationFunction;

            // 1. Calculate error values for last layer first.
            double[] layerOutputs = networkOutputs[layersCount - 1];
            double[] errors = networkErrors[layersCount - 1];

            for (int i = 0; i < errors.Length; i++)
            {
                double output = layerOutputs[i];
                double e = output - desiredOutput[i];
                errors[i] = e * function.Derivative2Function(output);
                sumOfSquaredErrors += e * e;
            }

            // 2. Calculate errors for all other layers
            for (int j = layersCount - 2; j >= 0; j--)
            {
                errors = networkErrors[j];
                layerOutputs = networkOutputs[j];

              
                var layerNext = Layers[j + 1] as ActivationLayerRProp;
                double[] nextErrors = networkErrors[j + 1];

                // For all neurons of this layer
                for (int i = 0; i < errors.Length; i++)
                {
                    double sum = 0.0;

                    // For all neurons of the next layer
                    for (int k = 0; k < nextErrors.Length; k++)
                        sum += nextErrors[k] * layerNext.Neurons[k].Weights[i];

                    errors[i] = sum * function.Derivative2Function(layerOutputs[i]);
                }
            }

            return sumOfSquaredErrors;
        }


        /// <summary>
        /// Calculate gradient
        /// </summary>
        /// <param name="input"></param>
        protected override void CalculateGradient(double[] input)
        {
            double[][] networkErrors = _networkErrors.Value;
            double[][] networkOutputs = _networkOutputs.Value;

            // 1. Calculate for last layer first
            double[] errors = networkErrors[0];
            double[][] layerWeightsDerivatives = WeightsDerivatives[0];
            double[] layerThresholdDerivatives = ThresholdsDerivatives[0];

            // For each neuron of the last layer
            for (int i = 0; i < errors.Length; i++)
            {
                double[] neuronWeightDerivatives = layerWeightsDerivatives[i];

                lock (neuronWeightDerivatives)
                {
                    // For each weight in the neuron
                    for (int j = 0; j < input.Length; j++)
                        neuronWeightDerivatives[j] += errors[i] * input[j];
                    layerThresholdDerivatives[i] += errors[i];
                }
            }

            // 2. Calculate for all other layers in a chain
            for (int k = 1; k < WeightsDerivatives.Length; k++)
            {
                errors = networkErrors[k];

                layerWeightsDerivatives = WeightsDerivatives[k];
                layerThresholdDerivatives = ThresholdsDerivatives[k];

                double[] layerPrev = networkOutputs[k - 1];

                // For each neuron in the current layer
                for (int i = 0; i < layerWeightsDerivatives.Length; i++)
                {
                    double[] neuronWeightDerivatives = layerWeightsDerivatives[i];

                    lock (neuronWeightDerivatives)
                    {
                        // For each weight of the neuron
                        for (int j = 0; j < neuronWeightDerivatives.Length; j++)
                            neuronWeightDerivatives[j] += errors[i] * layerPrev[j];
                        layerThresholdDerivatives[i] += errors[i];
                    }
                }
            }
        }
    }
}
