using System;
using NN.Eva.Core.ResilientPropagation.ActivationFunctions;
using NN.Eva.Services;

namespace NN.Eva.Core.ResilientPropagation
{
    public class NeuronRProp
    {
        /// <summary>
        /// Neuron's inputs count
        /// </summary>
        protected int _inputsCount = 0;

        /// <summary>
        /// Neuron's weights
        /// </summary>
        protected double[] _weights = null;

        /// <summary>
        /// Neuron's output value
        /// </summary>
        protected double _output = 0;


        public int InputsCount => _inputsCount;

        public double Output => _output;

        public double[] Weights
        {
            get => _weights;
            internal set => _weights = value;
        } 

        protected IActivationFunction _function = null;

        protected double _threshold = 0.0;

        public double Threshold
        {
            get => _threshold;
            internal set => _threshold = value;
        }

        /// <summary>
        /// Neuron's activation function.
        /// </summary>
        /// 
        public IActivationFunction ActivationFunction
        {
            get => _function;
            set => _function = value;
        }

        public NeuronRProp(int inputs, IActivationFunction function) : this(inputs) => _function = function;
       

        public NeuronRProp(int inputs)
        {
            _inputsCount = inputs;
            _weights = new double[_inputsCount];
        }


        public double Compute(double[] input)
        {
            // check for corrent input vector
            if (input.Length != _inputsCount)
                throw new ArgumentException("Wrong length of the input vector.");

            // initial sum value
            double sum = 0.0; int index = 0;
            foreach (var weight in _weights)
            {
                sum += weight * input[index];
                index++;
            }

            // compute weighted sum of inputs
            sum += _threshold;

            // local variable to avoid mutlithreaded conflicts
            double output = _function.ActivationFunction(sum);

            // assign output property as well (works correctly for single threaded usage)
            _output = output;

            return output;
        }


        public void SaveMemory(int layerNumber, int neuronNumber, string memoryPath)
        {
            double offsetValue = 1.0;
            FileManager.SaveMemory(layerNumber, neuronNumber, _weights, offsetValue, _threshold, memoryPath);
        }

    }
}
