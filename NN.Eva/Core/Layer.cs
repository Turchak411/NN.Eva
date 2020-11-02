using NN.Eva.Services;
using System;
using System.Collections.Generic;
using NN.Eva.Core.Database;
using NN.Eva.Models;

namespace NN.Eva.Core
{
    public class Layer
    {
        public Guid Id { get; set; } = Guid.NewGuid();

        private Neuron[] _neuronList;

        private Layer() { }

        public Layer(int neuronCount, int layerNumber)
        {
            List<Neuron> neuronList = new List<Neuron>();

            double offsetValue = 0.5;
            double offsetWeight = -1;

            for (int i = 0; i < neuronCount; i++)
            {
                double[] weights = FileManager.LoadMemory(layerNumber, i, ref offsetValue, ref offsetWeight);
                Neuron neuron = new Neuron(weights, offsetValue, offsetWeight, ActivationFunction.Sigmoid);

                neuronList.Add(neuron);
            }

            _neuronList = neuronList.ToArray();
        }

        public Layer(int neuronCount, int layerNumber, string memoryPath)
        {
            List<Neuron> neuronList = new List<Neuron>();

            double offsetValue = 0.5;
            double offsetWeight = -1;

            for (int i = 0; i < neuronCount; i++)
            {
                double[] weights = FileManager.LoadMemory(layerNumber, i, memoryPath, ref offsetValue, ref offsetWeight);
                Neuron neuron = new Neuron(weights, offsetValue, offsetWeight, ActivationFunction.Sigmoid);

                neuronList.Add(neuron);
            }

            _neuronList = neuronList.ToArray();
        }

        #region Handling

        public double[] Handle(double[] data)
        {
            double[] layerResultVector = new double[_neuronList.Length];

            int index = 0;

            foreach (double layerResult in layerResultVector)
            {
                layerResultVector[index] = _neuronList[index].Handle(data);
                index++;
            }

            return layerResultVector;
        }

        #endregion

        #region Teaching

        #region Error calculating

        public void CalcErrorAsOut(double[] rightAnswersSet)
        {
            int index = 0;

            foreach (Neuron neuron in _neuronList)
            {
                neuron.CalcErrorForOutNeuron(rightAnswersSet[index]);
                index++;
            }
        }

        public void CalcErrorAsHidden(double[][] nextLayerWeights, double[] nextLayerErrors)
        {
            int index = 0;

            foreach (Neuron neuron in _neuronList)
            {
                neuron.CalcErrorForHiddenNeuron(index, nextLayerWeights, nextLayerErrors);
                index++;
            }
        }

        #endregion

        #region Weights changing

        public void ChangeWeightsBProp(double learnSpeed, double[] answersFromPrevLayer)
        {
            foreach (Neuron neuron in _neuronList)
            {
                neuron.ChangeWeightsBProp(learnSpeed, answersFromPrevLayer);
            }
        }

        #endregion

        public double[] GetLastAnswers()
        {
            double[] lastAnswers = new double[_neuronList.Length];

            int index = 0;

            foreach (double lastAnswer in lastAnswers)
            {
                lastAnswers[index] = _neuronList[index].LastAnswer;
                index++;
            }

            return lastAnswers;
        }

        public double[][] GetWeights()
        {
            double[][] weights = new double[_neuronList.Length][];

            int index = 0;

            foreach (Neuron neuron in _neuronList)
            {
                weights[index] = neuron.Weights;
                index++;
            }

            return weights;
        }

        public double[] GetNeuronErrors()
        {
            double[] errors = new double[_neuronList.Length];

            int index = 0;

            foreach (Neuron neuron in _neuronList)
            {
                errors[index] = neuron.Error;
                index++;
            }

            return errors;
        }

        #endregion

        #region Memory operations

        public void SaveMemory(int layerNumber, string path)
        {
            for (int i = 0; i < _neuronList.Length; i++)
            {
                _neuronList[i].SaveMemory(layerNumber, i, path);
            }
        }

        #region Database operations

        public void SaveMemoryToDB(Guid networkId, Guid userId, int number, DBInserter dbInserter)
        {
            // Saving layer info:
            dbInserter.InsertLayer(Id, userId, networkId, number);

            // Saving neurons info:
            for (int i = 0; i < _neuronList.Length; i++)
            {
                _neuronList[i].SaveMemoryToDB(Id, userId, i, dbInserter);
            }
        }

        public void DBMemoryAbort(Guid networkId, DBDeleter dbDeleter)
        {
            // Aborting saving of neurons:
            for (int i = 0; i < _neuronList.Length; i++)
            {
                _neuronList[i].DBMemoryAbort(Id, dbDeleter);
            }

            // Aborting saving of layer:
            dbDeleter.DeleteFromTableLayers(networkId);
        }

        #endregion

        #endregion
    }
}
