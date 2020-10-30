using NN.Eva.Core.ResilientPropagation.ActivationFunctions;

namespace NN.Eva.Core.ResilientPropagation.Layers
{
    public class ActivationLayerRProp : LayerRProp
    {
        public ActivationLayerRProp(int neuronsCount, int inputsCount, IActivationFunction function)
            : base(neuronsCount, inputsCount)
        {
            // create each neuron
            for (int i = 0; i < _neurons.Length; i++)
                _neurons[i] = new NeuronRProp(inputsCount, function);
        }

        public void SetActivationFunction(IActivationFunction function)
        {
            foreach (var neuron in _neurons)
            {
                neuron.ActivationFunction = function;
            }
        }
    }
}
