using System;
using NN.Eva.Services;

namespace NN.Eva.Core.ResilientPropagation.Layers
{
    public class LayerRProp
    {
        protected int _inputsCount = 0;

        public int InputsCount => _inputsCount;

        protected int _neuronsCount = 0;

        public int NeuronsCount => _neuronsCount;

        protected NeuronRProp[] _neurons;

        public NeuronRProp[] Neurons => _neurons;

        protected double[] _output;

        public double[] Output => _output;

        private readonly object _lockLayer = new object();

        protected LayerRProp(int neuronsCount, int inputsCount)
        {
            _inputsCount = Math.Max(1, inputsCount);
            _neuronsCount = Math.Max(1, neuronsCount);
            _neurons = new NeuronRProp[_neuronsCount];
        }
        
        public double[] Compute(double[] input)
        {
            lock (_lockLayer)
            {
                // local variable to avoid mutlithread conflicts
                double[] output = new double[_neuronsCount];

                // compute each neuron
                for (int i = 0; i < _neurons.Length; i++)
                    output[i] = _neurons[i].Compute(input);

                // assign output property as well (works correctly for single threaded usage)
                _output = output;

                return output;
            }
        }

        public void LoadMemoryLayerRProp(int layerNumber)
        {
            double offsetValue = 0.5;
            double offsetWeight = -1;

            for (int i = 0; i < _neurons.Length; i++)
            {
                double[] weights = FileManager.LoadMemory(layerNumber, i,"memory.txt", ref offsetValue, ref offsetWeight);
                _neurons[i].Weights = weights;
                _neurons[i].Threshold = offsetWeight;
            }
        }

        public void SaveMemory(int layerNumber, string path)
        {
            for (int i = 0; i < _neurons.Length; i++)
            {
                _neurons[i].SaveMemory(layerNumber, i, path);
            }
        }
    }
}
