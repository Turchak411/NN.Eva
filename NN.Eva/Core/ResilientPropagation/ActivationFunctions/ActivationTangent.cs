using System;

namespace NN.Eva.Core.ResilientPropagation.ActivationFunctions
{
    /// <summary>
    /// Activation function Tangent
    /// Its output is in the range [-1, 1] and it is output.
    /// https://ru.wikipedia.org/wiki/Функция_активации
    /// </summary>
    public class ActivationTangent
    {
        public double ActivationFunction(double x) => (Math.Exp(2 * x) - 1) / (Math.Exp(2 * x) + 1);

        public double DerivativeFunction(double x) => 1 - Math.Sqrt(x);

        public double Derivative2Function(double y) => 1 - Math.Sqrt(y);
    }
}
