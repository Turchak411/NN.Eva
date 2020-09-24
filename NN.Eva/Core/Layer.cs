using NN.Eva.Services;
using System;
using System.Collections.Generic;
using System.Text;
using NN.Eva.Core.Database;
using NN.Eva.Models;

namespace NN.Eva.Core
{
    public class Layer
    {
        public Guid Id { get; set; } = Guid.NewGuid();

        private List<Neuron> _neuronList = new List<Neuron>();

        private Layer() { }

        public Layer(int neuronCount, int weightCount, int layerNumber, FileManager fileManager)
        {
            double offsetValue = 0.5;

            for (int i = 0; i < neuronCount; i++)
            {
                double[] weights = fileManager.LoadMemory(layerNumber, i);
                Neuron neuron = new Neuron(weights, offsetValue, -1, ActivationFunction.Sigmoid);

                _neuronList.Add(neuron);
            }
        }

        public Layer(int neuronCount, int weightCount, int layerNumber, FileManager fileManager, string memoryPath)
        {
            double offsetValue = 0.5;

            for (int i = 0; i < neuronCount; i++)
            {
                double[] weights = fileManager.LoadMemory(layerNumber, i, memoryPath);
                Neuron neuron = new Neuron(weights, offsetValue, -1, ActivationFunction.Sigmoid);

                _neuronList.Add(neuron);
            }
        }

        public double[] Handle(double[] data)
        {
            double[] layerResultVector = new double[_neuronList.Count];

            for (int i = 0; i < layerResultVector.Length; i++)
            {
                layerResultVector[i] = _neuronList[i].Handle(data);
            }

            return layerResultVector;
        }

        // CALCULATING ERRORS:

        public void CalcErrorAsOut(double[] rightAnwsersSet)
        {
            for (int i = 0; i < _neuronList.Count; i++)
            {
                _neuronList[i].CalcErrorForOutNeuron(rightAnwsersSet[i]);
            }
        }

        public void CalcErrorAsHidden(double[][] nextLayerWeights, double[] nextLayerErrors)
        {
            for (int i = 0; i < _neuronList.Count; i++)
            {
                _neuronList[i].CalcErrorForHiddenNeuron(i, nextLayerWeights, nextLayerErrors);
            }
        }

        // CHANGE WEIGHTS:

        public void ChangeWeights(double learnSpeed, double[] anwsersFromPrewLayer)
        {
            for (int i = 0; i < _neuronList.Count; i++)
            {
                _neuronList[i].ChangeWeights(learnSpeed, anwsersFromPrewLayer);
            }
        }

        public double[] GetLastAnwsers()
        {
            double[] lastAnwsers = new double[_neuronList.Count];

            for (int i = 0; i < _neuronList.Count; i++)
            {
                lastAnwsers[i] = _neuronList[i].GetLastAnwser();
            }

            return lastAnwsers;
        }

        public double[][] GetWeights()
        {
            double[][] weights = new double[_neuronList.Count][];

            for (int i = 0; i < _neuronList.Count; i++)
            {
                weights[i] = _neuronList[i].GetWeights();
            }

            return weights;
        }

        public double[] GetErrors()
        {
            double[] errors = new double[_neuronList.Count];

            for (int i = 0; i < _neuronList.Count; i++)
            {
                errors[i] = _neuronList[i].GetError();
            }

            return errors;
        }

        // SAVE MEMORY:

        public void SaveMemory(FileManager fileManager, int layerNumber, string path)
        {
            for (int i = 0; i < _neuronList.Count; i++)
            {
                _neuronList[i].SaveMemory(fileManager, layerNumber, i, path);
            }
        }

        public void SaveMemoryToDB(Guid networkId, Guid userId, int number, DBInserter dbInserter)
        {
            // Saving layer info:
            dbInserter.InsertLayer(Id, userId, networkId, number);

            // Saving neurons info:
            for (int i = 0; i < _neuronList.Count; i++)
            {
                _neuronList[i].SaveMemoryToDB(Id, userId, i, dbInserter);
            }
        }

        public void DBMemoryAbort(Guid networkId, DBDeleter dbDeleter)
        {
            // Aborting saving of neurons:
            for (int i = 0; i < _neuronList.Count; i++)
            {
                _neuronList[i].DBMemoryAbort(Id, dbDeleter);
            }

            // Aborting saving of layer:
            dbDeleter.DeleteFromTableLayers(networkId);
        }
    }
}
