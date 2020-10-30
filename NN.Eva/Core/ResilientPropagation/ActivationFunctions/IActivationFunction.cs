namespace NN.Eva.Core.ResilientPropagation.ActivationFunctions
{
    public interface IActivationFunction
    {
        double ActivationFunction(double x);

        double DerivativeFunction(double x);

        double Derivative2Function(double y);
    }
}
