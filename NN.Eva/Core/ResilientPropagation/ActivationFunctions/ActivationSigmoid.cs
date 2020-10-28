using System;

namespace NN.Eva.Core.ResilientPropagation.ActivationFunctions
{
    public class ActivationSigmoid : IActivationFunction
    {
        public double Alpha { get; set; } = 2;

        public double ActivationFunction(double x) => 1 / (1 + Math.Exp(-Alpha * x));

        public double DerivativeFunction(double x)
        {
            double y = ActivationFunction(x);
            return Alpha * y * (1 - y);
        }

        public double Derivative2Function(double y) => Alpha * y * (1 - y);
    }
}
