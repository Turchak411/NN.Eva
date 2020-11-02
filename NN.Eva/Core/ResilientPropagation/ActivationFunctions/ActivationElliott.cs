using System;

namespace NN.Eva.Core.ResilientPropagation.ActivationFunctions
{
    /// <summary>
    /// Alternative to ActivationSigmoid
    /// Its output is in the range [0, 1] and it is output.
    /// Elliott, D.L. "Improved activation function for artificial neural networks", 1993
    /// </summary>
    public class ActivationElliott : IActivationFunction
    {
        public double Alpha { get; set; } = 1;

        public double ActivationFunction(double x) => (x * Alpha) / 2 / (1 + Math.Abs(x * Alpha)) + 0.5;

        public double DerivativeFunction(double x) => Alpha / (2.0 * (1.0 + Math.Abs(x * Alpha)) * (1 + Math.Abs(x * Alpha)));

        public double Derivative2Function(double y) => Alpha / (2.0 * (1.0 + Math.Abs(y* Alpha)) * (1 + Math.Abs(y* Alpha)));
    }
}
