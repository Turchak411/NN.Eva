using System;
using System.Linq;
using NN.Eva.Core.ResilientPropagation.ActivationFunctions;
using NN.Eva.Core.ResilientPropagation.Layers;

namespace NN.Eva.Core.ResilientPropagation
{
    public class NeuralNetworkRProp
    {
        private LayerRProp[] _layers;

        private int _inputsCount;
        private double[] _output;

        private const double DeltaMax = 50.0;
        private const double DeltaMin = 1e-6;

        private const double EtaMinus = 0.5;
        private const double EtaPlus = 1.2;
        private const double LearningRate = 0.0125;
       

        private double[][] _neuronErrors;

        // update values, also known as deltas
        private double[][][] _weightsUpdates;
        private double[][] _thresholdsUpdates;

        // current and previous gradient values
        private double[][][] _weightsDerivatives;
        private double[][] _thresholdsDerivatives;

        private double[][][] _weightsPreviousDerivatives;
        private double[][] _thresholdsPreviousDerivatives;


        public NeuralNetworkRProp(IActivationFunction function, int inputsCount, params int[] neuronsCount) 
            : this(inputsCount, neuronsCount.Length)
        {
            for (int i = 0; i < _layers.Length; i++)
            {
                _layers[i] = new ActivationLayerRProp(neuronsCount[i],
                    i == 0 ? inputsCount : neuronsCount[i - 1], function);
            }
        }

        private NeuralNetworkRProp(int inputsCount, int layersCount)
        {
            _inputsCount = inputsCount;
            _layers = new LayerRProp[layersCount];
            InitializationNetwork(_layers.Length);
        }

        private void InitializationNetwork(int layersCount)
        {
            _neuronErrors = new double[layersCount][];

            _weightsDerivatives = new double[layersCount][][];
            _thresholdsDerivatives = new double[layersCount][];

            _weightsPreviousDerivatives = new double[layersCount][][];
            _thresholdsPreviousDerivatives = new double[layersCount][];

            _weightsUpdates = new double[layersCount][][];
            _thresholdsUpdates = new double[layersCount][];

            // initialize errors, derivatives and steps
            for (int i = 0; i < _layers.Length; i++)
            {
                LayerRProp layer = _layers[i];
                int neuronsCount = layer.Neurons.Length;

                _neuronErrors[i] = new double[neuronsCount];

                _weightsDerivatives[i] = new double[neuronsCount][];
                _weightsPreviousDerivatives[i] = new double[neuronsCount][];
                _weightsUpdates[i] = new double[neuronsCount][];

                _thresholdsDerivatives[i] = new double[neuronsCount];
                _thresholdsPreviousDerivatives[i] = new double[neuronsCount];
                _thresholdsUpdates[i] = new double[neuronsCount];

                // for each neuron
                for (int j = 0; j < layer.Neurons.Length; j++)
                {
                    _weightsDerivatives[i][j] = new double[layer.InputsCount];
                    _weightsPreviousDerivatives[i][j] = new double[layer.InputsCount];
                    _weightsUpdates[i][j] = new double[layer.InputsCount];
                }
            }

            // intialize steps
            ResetUpdates(LearningRate);
        }

        private void ResetUpdates(double rate)
        {
            foreach (var weight in _weightsUpdates)
            {
                for (int j = 0; j < weight.Length; j++)
                {
                    for (int k = 0; k < weight[j].Length; k++)
                    {
                        weight[j][k] = rate;
                    }
                }
            }

            foreach (var threshold in _thresholdsUpdates)
            {
                for (int j = 0; j < threshold.Length; j++)
                {
                    threshold[j] = rate;
                }
            }
        }

        public double Train(double[][] input, double[][] output)
        {
            ResetGradient();
            double error = 0.0;
            for (int i = 0; i < input.Length; i++)
            {
                // compute the network's output
                Compute(input[i]);

                // calculate network error
                error += CalculateError(input[i], output[i]);

                // calculate weights updates
                CalculateGradient(input[i]);
            }

            UpdateNetwork();

            return error / input.Length;
        }

        private double[] Compute(double[] input)
        {
            // local variable to avoid mutlithread conflicts
            double[] output = _layers.Aggregate(input, (current, layerRProp) => layerRProp.Compute(current));

            // assign output property as well (works correctly for single threaded usage)
            _output = output;

            return output;
        }

        private void ResetGradient()
        {
            foreach (var weightDerivative in _weightsDerivatives)
            {
                foreach (var weight in weightDerivative)
                {
                    Array.Clear(weight, 0, weight.Length);
                }
            }

            foreach (var threshold in _thresholdsDerivatives)
            {
                Array.Clear(threshold, 0, threshold.Length);
            }
        }


        private double CalculateError(double[] input, double[] desiredOutput)
        {
            double error = 0;
            int layersCount = _layers.Length;

            // assume, that all neurons of the network have the same activation function
            IActivationFunction function = _layers[0].Neurons[0].ActivationFunction;

            // calculate error values for the last layer first
            var layer = _layers[layersCount - 1] as ActivationLayerRProp;
            double[] layerDerivatives = _neuronErrors[layersCount - 1];

            for (int i = 0; i < layer?.Neurons.Length; i++)
            {
                double output = layer.Neurons[i].Output;

                double e = output - desiredOutput[i];
                layerDerivatives[i] = e * function.Derivative2Function(output);
                error += e * e;
            }


            // calculate error values for other layers
            for (int j = layersCount - 2; j >= 0; j--)
            {
                layer = _layers[j] as ActivationLayerRProp;
                layerDerivatives = _neuronErrors[j];

                var layerNext = _layers[j + 1] as ActivationLayerRProp;
                double[] nextDerivatives = _neuronErrors[j + 1];

                // for all neurons of the layer
                for (int i = 0, n = layer.Neurons.Length; i < n; i++)
                {
                    var sum = layerNext.Neurons.Select((neuronRProp, k) => nextDerivatives[k] * layerNext.Neurons[k].Weights[i]).Sum();

                    layerDerivatives[i] = sum * function.Derivative2Function(layer.Neurons[i].Output);
                }
            }

            return error/2.0;
        }

        private void UpdateNetwork()
        {
            double[][] layerWeightsUpdates;
            double[] layerThresholdUpdates;
            double[] neuronWeightUpdates;

            double[][] layerWeightsDerivatives;
            double[] layerThresholdDerivatives;
            double[] neuronWeightDerivatives;

            double[][] layerPreviousWeightsDerivatives;
            double[] layerPreviousThresholdDerivatives;
            double[] neuronPreviousWeightDerivatives;

            // for each layer of the network
            for (int i = 0; i < _layers.Length; i++)
            {
                var layer = _layers[i] as ActivationLayerRProp;

                layerWeightsUpdates = _weightsUpdates[i];
                layerThresholdUpdates = _thresholdsUpdates[i];

                layerWeightsDerivatives = _weightsDerivatives[i];
                layerThresholdDerivatives = _thresholdsDerivatives[i];

                layerPreviousWeightsDerivatives = _weightsPreviousDerivatives[i];
                layerPreviousThresholdDerivatives = _thresholdsPreviousDerivatives[i];

                // for each neuron of the layer
                for (int j = 0; j < layer.Neurons.Length; j++)
                {
                    var neuron = layer.Neurons[j];

                    neuronWeightUpdates = layerWeightsUpdates[j];
                    neuronWeightDerivatives = layerWeightsDerivatives[j];
                    neuronPreviousWeightDerivatives = layerPreviousWeightsDerivatives[j];

                    double change = 0;

                    // for each weight of the neuron
                    for (int k = 0; k < neuron.InputsCount; k++)
                    {
                        change = Sign(neuronPreviousWeightDerivatives[k] * neuronWeightDerivatives[k]);

                        if (change > 0)
                        {
                            double delta = neuronWeightUpdates[k] * EtaPlus;
                            delta = Math.Min(delta, DeltaMax);
                            neuronWeightUpdates[k] = delta;
                            neuron.Weights[k] -= Sign(neuronWeightDerivatives[k]) * delta;
                            neuronPreviousWeightDerivatives[k] = neuronWeightDerivatives[k];
                        }
                        else if (change < 0)
                        {
                            double delta = neuronWeightUpdates[k] * EtaMinus;
                            delta = Math.Max(delta, DeltaMin);
                            neuronWeightUpdates[k] = delta;
                            neuronPreviousWeightDerivatives[k] = 0;
                        }
                        else if (change == 0)
                        {
                            double delta = neuronWeightUpdates[k];
                            neuron.Weights[k] -= Sign(neuronWeightDerivatives[k]) * delta;
                            neuronPreviousWeightDerivatives[k] = neuronWeightDerivatives[k];
                        }
                    }

                    // update treshold
                    change = Sign(layerPreviousThresholdDerivatives[j] * layerThresholdDerivatives[j]);

                    if (change > 0)
                    {
                        layerThresholdUpdates[j] = Math.Min(layerThresholdUpdates[j] * EtaPlus, DeltaMax);
                        neuron.Threshold -= Sign(layerThresholdDerivatives[j]) * layerThresholdUpdates[j];
                        layerPreviousThresholdDerivatives[j] = layerThresholdDerivatives[j];
                    }
                    else if (change < 0)
                    {
                        layerThresholdUpdates[j] = Math.Max(layerThresholdUpdates[j] * EtaMinus, DeltaMin);
                        layerThresholdDerivatives[j] = 0;
                    }
                    else if (change == 0)
                    {
                        neuron.Threshold -= Sign(layerThresholdDerivatives[j]) * layerThresholdUpdates[j];
                        layerPreviousThresholdDerivatives[j] = layerThresholdDerivatives[j];
                    }
                }
            }
        }


        private void CalculateGradient(double[] input)
        {
            // 1. calculate updates for the first layer
            var layer = _layers[0] as ActivationLayerRProp;
            double[] weightErrors = _neuronErrors[0];
            double[][] layerWeightsDerivatives = _weightsDerivatives[0];
            double[] layerThresholdDerivatives = _thresholdsDerivatives[0];

            // So, for each neuron of the first layer:
            for (int i = 0; i < layer?.Neurons.Length; i++)
            {
                var neuron = layer.Neurons[i];
                double[] neuronWeightDerivatives = layerWeightsDerivatives[i];

                // for each weight of the neuron:
                for (int j = 0; j < neuron.InputsCount; j++)
                {
                    neuronWeightDerivatives[j] += weightErrors[i] * input[j];
                }
                layerThresholdDerivatives[i] += weightErrors[i];
            }

            // 2. for all other layers
            for (int k = 1; k < _layers.Length; k++)
            {
                layer = _layers[k] as ActivationLayerRProp;
                weightErrors = _neuronErrors[k];
                layerWeightsDerivatives = _weightsDerivatives[k];
                layerThresholdDerivatives = _thresholdsDerivatives[k];

                var layerPrev = _layers[k - 1] as ActivationLayerRProp;

                // for each neuron of the layer
                for (int i = 0; i < layer?.Neurons.Length; i++)
                {
                    double[] neuronWeightDerivatives = layerWeightsDerivatives[i];

                    // for each weight of the neuron
                    for (int j = 0; j < layerPrev?.Neurons.Length; j++)
                    {
                        neuronWeightDerivatives[j] += weightErrors[i] * layerPrev.Neurons[j].Output;
                    }
                    layerThresholdDerivatives[i] += weightErrors[i];
                }
            }
        }

        private int Sign(double v)
        {
            if (Math.Abs(v) < 0.00000000000000001) return 0;
            if (v > 0) return 1;
            return -1;
        }
    }
}
