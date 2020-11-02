namespace NN.Eva.Core.ResilientPropagation.ActivationFunctions
{
    public class ActivationLinear : IActivationFunction
    {
        public double ActivationFunction(double x) => x;

        public double DerivativeFunction(double x) => 1;

        public double Derivative2Function(double y) => 1;
    }
}
