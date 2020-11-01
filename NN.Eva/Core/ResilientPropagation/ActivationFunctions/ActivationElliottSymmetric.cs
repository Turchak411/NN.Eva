using System;

namespace NN.Eva.Core.ResilientPropagation.ActivationFunctions
{
    /// <summary>
    /// Computationally efficient alternative to ActivationTANH.
    /// Its output is in the range [-1, 1], and it is derivable.
    /// Elliott, D.L. "A better activation function for artificial neural networks", 1993
    /// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.46.7204&rep=rep1&type=pdf
    /// </summary>
    public class ActivationElliottSymmetric : IActivationFunction
    {
        public double Alpha { get; set; } = 1;

        public double ActivationFunction(double x) => (x * Alpha) / (1 + Math.Abs(x * Alpha));

        public double DerivativeFunction(double x)
        {
            var denominator = 1.0 + Math.Abs(x * Alpha);
            return (Alpha * 1.0) / (denominator * denominator);
        }

        public double Derivative2Function(double y)
        {
            var denominator = 1.0 + Math.Abs(y * Alpha);
            return (Alpha * 1.0) / (denominator * denominator);
        }
    }
}
