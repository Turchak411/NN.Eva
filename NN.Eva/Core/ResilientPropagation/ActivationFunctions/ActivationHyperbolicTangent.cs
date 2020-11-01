using System;

namespace NN.Eva.Core.ResilientPropagation.ActivationFunctions
{
    /// <summary>
    /// The hyperbolic tangent activation function
    /// https://neurohive.io/ru/osnovy-data-science/activation-functions/
    /// </summary>
    public class ActivationHyperbolicTangent : IActivationFunction
    {
        public double ActivationFunction(double x) => 2.0 / (1.0 + Math.Exp(-2.0 * x)) - 1.0;

        public double DerivativeFunction(double x) => 1.0d - x * x;

        public double Derivative2Function(double y) => 1.0d - y * y;
    }
}
