using System;
using System.Linq;
using NN.Eva.Models;
using NN.Eva.Services;
using NN.Eva.Core.ResilientPropagation.Layers;
using NN.Eva.Core.ResilientPropagation.ActivationFunctions;

namespace NN.Eva.Core.ResilientPropagation
{
    public class NeuralNetworkRProp
    {
        protected readonly LayerRProp[] Layers;

        private const double DeltaMax = 50.0;
        private const double DeltaMin = 1e-6;

        private const double EtaMinus = 0.5;
        private const double EtaPlus = 1.2;
        private const double LearningRate = 0.0125;
        
        private double[][] _neuronErrors;

        // update values, also known as deltas
        protected double[][][] WeightsUpdates;
        protected double[][] ThresholdsUpdates;

        // current and previous gradient values
        protected double[][][] WeightsDerivatives;
        protected double[][] ThresholdsDerivatives;

        protected double[][][] WeightsPreviousDerivatives;
        protected double[][] ThresholdsPreviousDerivatives;


        public NeuralNetworkRProp(IActivationFunction function, NetworkStructure networkStructure)
        {
            Layers = new LayerRProp[networkStructure.NeuronsByLayers.Length];

            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i] = new ActivationLayerRProp(networkStructure.NeuronsByLayers[i],
                    i == 0 ? networkStructure.InputVectorLength : networkStructure.NeuronsByLayers[i - 1], function);
                Layers[i].LoadMemoryLayerRProp(i);
            }

            InitializationNetwork(Layers.Length);
        }


        private void InitializationNetwork(int layersCount)
        {
            _neuronErrors = new double[layersCount][];

            WeightsDerivatives = new double[layersCount][][];
            ThresholdsDerivatives = new double[layersCount][];

            WeightsPreviousDerivatives = new double[layersCount][][];
            ThresholdsPreviousDerivatives = new double[layersCount][];

            WeightsUpdates = new double[layersCount][][];
            ThresholdsUpdates = new double[layersCount][];

            // initialize errors, derivatives and steps
            for (int i = 0; i < Layers.Length; i++)
            {
                LayerRProp layer = Layers[i];
                int neuronsCount = layer.Neurons.Length;

                _neuronErrors[i] = new double[neuronsCount];

                WeightsDerivatives[i] = new double[neuronsCount][];
                WeightsPreviousDerivatives[i] = new double[neuronsCount][];
                WeightsUpdates[i] = new double[neuronsCount][];

                ThresholdsDerivatives[i] = new double[neuronsCount];
                ThresholdsPreviousDerivatives[i] = new double[neuronsCount];
                ThresholdsUpdates[i] = new double[neuronsCount];

                // for each neuron
                for (int j = 0; j < layer.Neurons.Length; j++)
                {
                    WeightsDerivatives[i][j] = new double[layer.InputsCount];
                    WeightsPreviousDerivatives[i][j] = new double[layer.InputsCount];
                    WeightsUpdates[i][j] = new double[layer.InputsCount];
                }
            }

            // intialize steps
            ResetUpdates(LearningRate);
        }

        private void ResetUpdates(double rate)
        {
            foreach (var weight in WeightsUpdates)
            {
                for (int j = 0; j < weight.Length; j++)
                {
                    for (int k = 0; k < weight[j].Length; k++)
                    {
                        weight[j][k] = rate;
                    }
                }
            }

            foreach (var threshold in ThresholdsUpdates)
            {
                for (int j = 0; j < threshold.Length; j++)
                {
                    threshold[j] = rate;
                }
            }
        }

        public virtual double Train(double[][] input, double[][] output)
        {
            ResetGradient();
            double error = 0.0;
            for (int i = 0; i < input.Length; i++)
            {
                // compute the network's output
                Compute(input[i]);

                // calculate network error
                error += CalculateError(output[i]);

                // calculate weights updates
                CalculateGradient(input[i]);
            }

            UpdateNetwork();

            return error / input.Length;
        }

        public double[] Compute(double[] input) => 
            Layers.Aggregate(input, (current, layerRProp) => layerRProp.Compute(current));

        protected virtual void ResetGradient()
        {
            foreach (var weightDerivative in WeightsDerivatives)
            {
                foreach (var weight in weightDerivative)
                {
                    Array.Clear(weight, 0, weight.Length);
                }
            }

            foreach (var threshold in ThresholdsDerivatives)
            {
                Array.Clear(threshold, 0, threshold.Length);
            }
        }

        protected virtual double CalculateError(double[] desiredOutput)
        {
            double error = 0;
            int layersCount = Layers.Length;

            // assume, that all neurons of the network have the same activation function
            IActivationFunction function = Layers[0].Neurons[0].ActivationFunction;

            // calculate error values for the last layer first
            var layer = Layers[layersCount - 1] as ActivationLayerRProp;
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
                layer = Layers[j] as ActivationLayerRProp;
                layerDerivatives = _neuronErrors[j];

                var layerNext = Layers[j + 1] as ActivationLayerRProp;
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

        protected void UpdateNetwork()
        {
            // for each layer of the network
            for (int i = 0; i < Layers.Length; i++)
            {
                var layer = Layers[i] as ActivationLayerRProp;

                var layerWeightsUpdates = WeightsUpdates[i];
                var layerThresholdUpdates = ThresholdsUpdates[i];

                var layerWeightsDerivatives = WeightsDerivatives[i];
                var layerThresholdDerivatives = ThresholdsDerivatives[i];

                var layerPreviousWeightsDerivatives = WeightsPreviousDerivatives[i];
                var layerPreviousThresholdDerivatives = ThresholdsPreviousDerivatives[i];

                // for each neuron of the layer
                for (int layerNeuronIndex = 0; layerNeuronIndex < layer?.Neurons.Length; layerNeuronIndex++)
                {
                    var neuron = layer.Neurons[layerNeuronIndex];

                    var neuronWeightUpdates = layerWeightsUpdates[layerNeuronIndex];
                    var neuronWeightDerivatives = layerWeightsDerivatives[layerNeuronIndex];
                    var neuronPreviousWeightDerivatives = layerPreviousWeightsDerivatives[layerNeuronIndex];

                    // for each weight of the neuron
                    for (int index = 0; index < neuron.InputsCount; index++)
                    {
                        UpdatedWeightPlus(neuronPreviousWeightDerivatives, neuronWeightDerivatives, 
                            neuronWeightUpdates, neuron, index);
                    }

                    UpdateThresholdPlus(layerPreviousThresholdDerivatives, layerThresholdDerivatives,
                        layerThresholdUpdates, neuron, layerNeuronIndex);
                }
            }
        }

        private void UpdatedWeightPlus(double[] neuronPreviousWeightDerivatives, double[] neuronWeightDerivatives, 
            double[] neuronWeightUpdates, NeuronRProp neuron, int index)
        {
            int change = Sign(neuronPreviousWeightDerivatives[index] * neuronWeightDerivatives[index]);

            if (change > 0)
            {
                double delta = neuronWeightUpdates[index] * EtaPlus;
                delta = Math.Min(delta, DeltaMax);
                neuronWeightUpdates[index] = delta;
                neuron.Weights[index] -= Sign(neuronWeightDerivatives[index]) * delta;
                neuronPreviousWeightDerivatives[index] = neuronWeightDerivatives[index];
            }
            else if (change < 0)
            {
                double delta = neuronWeightUpdates[index] * EtaMinus;
                delta = Math.Max(delta, DeltaMin);
                neuronWeightUpdates[index] = delta;
                neuronPreviousWeightDerivatives[index] = 0;
            }
            else if (change == 0)
            {
                double delta = neuronWeightUpdates[index];
                neuron.Weights[index] -= Sign(neuronWeightDerivatives[index]) * delta;
                neuronPreviousWeightDerivatives[index] = neuronWeightDerivatives[index];
            }
        }

        private void UpdateThresholdPlus(double[] layerPreviousThresholdDerivatives, double[] layerThresholdDerivatives,
            double[] layerThresholdUpdates, NeuronRProp neuron, int index)
        {
            int change = Sign(layerPreviousThresholdDerivatives[index] * layerThresholdDerivatives[index]);

            if (change > 0)
            {
                layerThresholdUpdates[index] = Math.Min(layerThresholdUpdates[index] * EtaPlus, DeltaMax);
                neuron.Threshold -= Sign(layerThresholdDerivatives[index]) * layerThresholdUpdates[index];
                layerPreviousThresholdDerivatives[index] = layerThresholdDerivatives[index];
            }
            else if (change < 0)
            {
                layerThresholdUpdates[index] = Math.Max(layerThresholdUpdates[index] * EtaMinus, DeltaMin);
                layerThresholdDerivatives[index] = 0;
            }
            else if (change == 0)
            {
                neuron.Threshold -= Sign(layerThresholdDerivatives[index]) * layerThresholdUpdates[index];
                layerPreviousThresholdDerivatives[index] = layerThresholdDerivatives[index];
            }
        }

        protected virtual void CalculateGradient(double[] input)
        {
            // 1. calculate updates for the first layer
            var layer = Layers[0] as ActivationLayerRProp;
            double[] weightErrors = _neuronErrors[0];
            double[][] layerWeightsDerivatives = WeightsDerivatives[0];
            double[] layerThresholdDerivatives = ThresholdsDerivatives[0];

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
            for (int k = 1; k < Layers.Length; k++)
            {
                layer = Layers[k] as ActivationLayerRProp;
                weightErrors = _neuronErrors[k];
                layerWeightsDerivatives = WeightsDerivatives[k];
                layerThresholdDerivatives = ThresholdsDerivatives[k];

                var layerPrev = Layers[k - 1] as ActivationLayerRProp;

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

        public void SaveMemory(string path, NetworkStructure networkStructure)
        {
            // Deleting old memory file:
            FileManager.PrepareToSaveMemory(path, networkStructure);

            // Saving
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].SaveMemory(i, path);
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
