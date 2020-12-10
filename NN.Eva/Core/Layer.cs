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

        public Layer(int neuronCount, int layerNumber, double alpha)
        {
            List<Neuron> neuronList = new List<Neuron>();

            double offsetValue = 0.5;
            double offsetWeight = -1;

            for (int i = 0; i < neuronCount; i++)
            {
                double[] weights = FileManager.LoadMemory(layerNumber, i, ref offsetValue, ref offsetWeight);
                Neuron neuron = new Neuron(weights, offsetValue, offsetWeight, ActivationFunction.Sigmoid, alpha);

                neuronList.Add(neuron);
            }

            _neuronList = neuronList.ToArray();
        }

        public Layer(int neuronCount, int layerNumber, string memoryPath, double alpha)
        {
            List<Neuron> neuronList = new List<Neuron>();

            double offsetValue = 0.5;
            double offsetWeight = -1;

            for (int i = 0; i < neuronCount; i++)
            {
                double[] weights = FileManager.LoadMemory(layerNumber, i, memoryPath, ref offsetValue, ref offsetWeight);
                Neuron neuron = new Neuron(weights, offsetValue, offsetWeight, ActivationFunction.Sigmoid, alpha);

                neuronList.Add(neuron);
            }

            _neuronList = neuronList.ToArray();
        }

        #region Handling

        public double[] Handle(double[] data)
        {
            double[] layerResultVector = new double[_neuronList.Length];

            int foreachIndex = 0;

            foreach (double layerResult in layerResultVector)
            {
                layerResultVector[foreachIndex] = _neuronList[foreachIndex].Handle(data);
                foreachIndex++;
            }

            return layerResultVector;
        }

        #endregion

        #region Teaching

        #region Error calculating

        public void CalcErrorAsOut(double[] rightAnswersSet)
        {
            int foreachIndex = 0;

            foreach (Neuron neuron in _neuronList)
            {
                neuron.CalcErrorForOutNeuron(rightAnswersSet[foreachIndex]);
                foreachIndex++;
            }
        }

        public void CalcErrorAsHidden(double[][] nextLayerWeights, double[] nextLayerErrors)
        {
            int foreachIndex = 0;

            foreach (Neuron neuron in _neuronList)
            {
                neuron.CalcErrorForHiddenNeuron(foreachIndex, nextLayerWeights, nextLayerErrors);
                foreachIndex++;
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

            int foreachIndex = 0;

            foreach (double lastAnswer in lastAnswers)
            {
                lastAnswers[foreachIndex] = _neuronList[foreachIndex].LastAnswer;
                foreachIndex++;
            }

            return lastAnswers;
        }

        public double[][] GetWeights()
        {
            double[][] weights = new double[_neuronList.Length][];

            int foreachIndex = 0;

            foreach (Neuron neuron in _neuronList)
            {
                weights[foreachIndex] = neuron.Weights;
                foreachIndex++;
            }

            return weights;
        }

        public double[] GetNeuronErrors()
        {
            double[] errors = new double[_neuronList.Length];

            int foreachIndex = 0;

            foreach (Neuron neuron in _neuronList)
            {
                errors[foreachIndex] = neuron.Error;
                foreachIndex++;
            }

            return errors;
        }

        #endregion

        #region Memory operations

        public void SaveMemory(int layerNumber, string path)
        {
            int foreachIndex = 0;

            foreach (Neuron neuron in _neuronList)
            {
                neuron.SaveMemory(layerNumber, foreachIndex, path);
                foreachIndex++;
            }
        }

        #region Database operations

        public void SaveMemoryToDB(Guid networkId, Guid userId, int number, DBInserter dbInserter)
        {
            // Saving layer info:
            dbInserter.InsertLayer(Id, userId, networkId, number);

            // Saving neurons info:
            int foreachIndex = 0;

            foreach (Neuron neuron in _neuronList)
            {
                neuron.SaveMemoryToDB(Id, userId, foreachIndex, dbInserter);
                foreachIndex++;
            }
        }

        public void DBMemoryAbort(Guid networkId, DBDeleter dbDeleter)
        {
            // Aborting saving of neurons:
            foreach (Neuron neuron in _neuronList)
            {
                neuron.DBMemoryAbort(Id, dbDeleter);
            }

            // Aborting saving of layer:
            dbDeleter.DeleteFromTableLayers(networkId);
        }

        #endregion

        #endregion
    }
}
