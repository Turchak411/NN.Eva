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

        private double _lastAnwser;

        private double _error;

        private ActivationFunction _actFunc;

        private Neuron() { }

        public Neuron(double[] weightsValues, double offsetValue, double offsetWeight, ActivationFunction actFunc)
        {
            _weights = weightsValues;
            _offsetValue = offsetValue;
            _offsetWeight = offsetWeight;

            _error = 1;
            _actFunc = actFunc;
        }

        public double Handle(double[] data)
        {
            double x = CalcSum(data);
            double actFunc = ActivationFunction(x);

            _lastAnwser = actFunc;
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
            switch (_actFunc)
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

        // CALCULATING ERRORS:

        public void CalcErrorForOutNeuron(double rightAnwser)
        {
            _error = (rightAnwser - _lastAnwser) * _lastAnwser * (1 - _lastAnwser);
        }

        public double CalcErrorForHiddenNeuron(int neuronIndex, double[][] nextLayerWeights, double[] nextLayerErrors)
        {
            // Вычисление производной активационной функции:
            _error = _lastAnwser * (1 - _lastAnwser);

            // Суммирование ошибок со следующего слоя:
            double sum = 0;

            for (int i = 0; i < nextLayerWeights.GetLength(0); i++)
            {
                sum += nextLayerWeights[i][neuronIndex] * nextLayerErrors[i];
            }

            _error = _error * sum;

            return _error;
        }

        public double[] GetWeights()
        {
            return _weights;
        }

        public double GetError()
        {
            return _error;
        }

        // CHANGE WEIGHTS:

        public void ChangeWeights(double learnSpeed, double[] anwsersFromPrewLayer)
        {
            for (int i = 0; i < _weights.Length; i++)
            {
                _weights[i] = _weights[i] + learnSpeed * _error * anwsersFromPrewLayer[i];
            }

            // Изменение величины смещения:
            _offsetWeight = _offsetWeight + learnSpeed * _error;
        }

        public double GetLastAnwser()
        {
            return _lastAnwser;
        }

        // SAVE MEMORY:

        public void SaveMemory(FileManager fileManager, int layerNumber, int neuronNumber, string memoryPath)
        {
            fileManager.SaveMemory(layerNumber, neuronNumber, _weights, _offsetValue, _offsetWeight, memoryPath);
        }

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

        // MEMORY:

        public bool IsMemoryEquals(int weightsCount)
        {
            return _weights.Length == weightsCount;
        }
    }
}
