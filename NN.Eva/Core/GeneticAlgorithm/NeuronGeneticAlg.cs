using System;
using NN.Eva.Models;

namespace NN.Eva.Core.GeneticAlgorithm
{
    public class NeuronGeneticAlg
    {
        public double[] Weights { get; set; }

        public double OffsetValue { get; set; }

        public double OffsetWeight { get; set; }

        public ActivationFunction ActivationFunctionType { get; set; }

        public double Handle(double[] data)
        {
            double x = CalcSum(data);
            double actFunc = ActivationFunction(x);

            return actFunc;
        }

        private double CalcSum(double[] data)
        {
            double x = 0;

            for (int i = 0; i < Weights.Length; i++)
            {
                x += Weights[i] * data[i];
            }

            return x + OffsetValue * OffsetWeight;
        }

        private double ActivationFunction(double x)
        {
            switch (ActivationFunctionType)
            {
                case Models.ActivationFunction.Th:
                    return (Math.Exp(2 * x) - 1) / (Math.Exp(2 * x) + 1);
                case Models.ActivationFunction.SoftPlus:
                    return Math.Log(1 + Math.Exp(x));
                case Models.ActivationFunction.Sigmoid:
                default:
                    return 1 / (1 + Math.Exp(-x));
            }
        }
    }
}
