using System;

namespace NN.Eva.Core.ResilientPropagation.ActivationFunctions
{
    /// <summary>
    /// Activation function SoftPlus
    /// Its output is in the range [0, +] and it is output.
    /// https://ru.wikipedia.org/wiki/Функция_активации
    /// </summary>
    public class ActivationSoftPlus : IActivationFunction
    {
        public double ActivationFunction(double x) => Math.Log(1 + Math.Exp(x));

        public double DerivativeFunction(double x) => 1 / (1 + Math.Exp(-x));

        public double Derivative2Function(double y) => 1 / (1 + Math.Exp(-y));
    }
}
