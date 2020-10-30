using System;
using NN.Eva.Core.Database;
using NN.Eva.Models;
using NN.Eva.Services;

namespace NN.Eva.Core
{
    public class Neuron
    {
        public Guid Id { get; set; } = Guid.NewGuid();

        private double[] _weights;

        private double _offsetValue;
        private double _offsetWeight;

        private double _lastAnswer;

        private double _error;

        private ActivationFunction _activationFunctionType;

        #region Out properties

        public double Error => _error;

        public double[] Weights => _weights;

        public double LastAnswer => _lastAnswer;

        #endregion

        private Neuron() { }

        public Neuron(double[] weightsValues, double offsetValue, double offsetWeight, ActivationFunction activationFunctionType)
        {
            _weights = weightsValues;
            _offsetValue = offsetValue;
            _offsetWeight = offsetWeight;

            _error = 1;
            _activationFunctionType = activationFunctionType;
        }

        #region Handling

        public double Handle(double[] data)
        {
            double x = CalcSum(data);
            double actFunc = ActivationFunction(x);

            _lastAnswer = actFunc;
            return actFunc;
        }

        private double CalcSum(double[] data)
        {
            double x = 0;

            for (int i = 0; i < _weights.Length; i++)
            {
                x += _weights[i] * data[i];
            }

            return x + _offsetValue * _offsetWeight;
        }

        private double ActivationFunction(double x)
        {
            switch (_activationFunctionType)
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

        #endregion

        #region Teaching

        #region Error calculating

        public void CalcErrorForOutNeuron(double rightAnswer)
        {
            _error = (rightAnswer - _lastAnswer) * _lastAnswer * (1 - _lastAnswer);
        }

        public double CalcErrorForHiddenNeuron(int neuronIndex, double[][] nextLayerWeights, double[] nextLayerErrors)
        {
            // Вычисление производной активационной функции:
            _error = _lastAnswer * (1 - _lastAnswer);

            // Суммирование ошибок со следующего слоя:
            double sum = 0;

            for (int i = 0; i < nextLayerWeights.GetLength(0); i++)
            {
                sum += nextLayerWeights[i][neuronIndex] * nextLayerErrors[i];
            }

            _error = _error * sum;

            return _error;
        }

        #endregion

        #region Weights changing

        public void ChangeWeightsBProp(double learnSpeed, double[] answersFromPrevLayer)
        {
            for (int i = 0; i < _weights.Length; i++)
            {
                _weights[i] = _weights[i] + learnSpeed * _error * answersFromPrevLayer[i];
            }

            // Изменение величины смещения:
            _offsetWeight = _offsetWeight + learnSpeed * _error;
        }

        public void ChangeWeightsRProp(double updateValue)
        {
            for (int i = 0; i < _weights.Length; i++)
            {
                _weights[i] = _weights[i] + updateValue;
            }
        }

        #endregion

        #endregion

        #region Memory operations

        public void SaveMemory(int layerNumber, int neuronNumber, string memoryPath)
        {
            FileManager.SaveMemory(layerNumber, neuronNumber, _weights, _offsetValue, _offsetWeight, memoryPath);
        }

        #region Database operations

        public void SaveMemoryToDB(Guid layerId, Guid userId, int number, DBInserter dbInserter)
        {
            // Saving networks info:
            dbInserter.InsertNeuron(Id, userId, layerId, number, _offsetValue, _offsetWeight);

            // Saving weights info:
            for (int i = 0; i < _weights.Length; i++)
            {
                dbInserter.InsertWeight(Id, userId, i, _weights[i]);
            }
        }

        public void DBMemoryAbort(Guid layerId, DBDeleter dbDeleter)
        {
            // Aborting saving of neurons:
            for (int i = 0; i < _weights.Length; i++)
            {
                dbDeleter.DeleteFromTableWeights(Id);
            }

            // Aborting saving of layer:
            dbDeleter.DeleteFromTableNeurons(layerId);
        }

        #endregion

        #endregion
    }
}
