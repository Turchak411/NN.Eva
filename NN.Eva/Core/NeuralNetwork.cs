using System;
using System.Collections.Generic;
using NN.Eva.Core.Database;
using NN.Eva.Models;
using NN.Eva.Services;

namespace NN.Eva.Core
{
    public class NeuralNetwork
    {
        public Guid Id { get; set; } = Guid.NewGuid();

        protected Layer[] _layerList;

        protected NeuralNetwork() { }

        public NeuralNetwork(int[] neuronsNumberByLayers)
        {
            List<Layer> layerList = new List<Layer>();

            Layer firstLayer = new Layer(neuronsNumberByLayers[0], 0);
            layerList.Add(firstLayer);

            for (int i = 1; i < neuronsNumberByLayers.Length; i++)
            {
                Layer layer = new Layer(neuronsNumberByLayers[i], i);
                layerList.Add(layer);
            }

            _layerList = layerList.ToArray();
        }

        public NeuralNetwork(int[] neuronsNumberByLayers, string memoryPath)
        {
            List<Layer> layerList = new List<Layer>();

            Layer firstLayer = new Layer(neuronsNumberByLayers[0], 0, memoryPath);
            layerList.Add(firstLayer);

            for (int i = 1; i < neuronsNumberByLayers.Length; i++)
            {
                Layer layer = new Layer(neuronsNumberByLayers[i], i, memoryPath);
                layerList.Add(layer);
            }

            _layerList = layerList.ToArray();
        }

        #region Handling

        /// <summary>
        /// Handling data
        /// </summary>
        /// <param name="data"></param>
        /// <param name="errorText"></param>
        /// <returns></returns>
        public double[] Handle(double[] data, ref string errorText)
        {
            // Check for non equaling of input length of data and network's receptors:
            if (data.Length != _layerList[0].GetWeights()[0].Length)
            {
                errorText = String.Format("Expected by network input-vector length: {0}\nInputed data-vector length: {1}", _layerList[0].GetWeights()[0].Length, data.Length);
                return null;
            }

            double[] tempData = data;

            foreach(Layer layer in _layerList)
            {
                tempData = layer.Handle(tempData);
            }

            // There is one double value at the last handle
            return tempData;
        }

        /// <summary>
        /// Handling data without vector's length check
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public double[] HandleUnsafe(double[] data)
        {
            double[] tempData = data;

            foreach (Layer layer in _layerList)
            {
                tempData = layer.Handle(tempData);
            }

            // There is one double value at the last handle
            return tempData;
        }

        #endregion

        #region Teaching

        public void CalculateErrors(double[] rightAnswersSet)
        {
            _layerList[_layerList.Length - 1].CalcErrorAsOut(rightAnswersSet);

            for (int i = _layerList.Length - 2; i >= 0; i--)
            {
                double[][] nextLayerWeights = _layerList[i + 1].GetWeights();
                double[] nextLayerErrors = _layerList[i + 1].GetNeuronErrors();

                _layerList[i].CalcErrorAsHidden(nextLayerWeights, nextLayerErrors);
            }
        }

        public void TeachBProp(double[] data, double[] rightAnswersSet, double learnSpeed)
        {
            // Подсчет ошибки (внутреннее изменение):
            CalculateErrors(rightAnswersSet);

            // Корректировка весов нейронов:
            double[] answersFromPrevLayer = data;

            foreach (Layer layer in _layerList)
            {
                layer.ChangeWeightsBProp(learnSpeed, answersFromPrevLayer);
                answersFromPrevLayer = layer.GetLastAnswers();
            }
        }

        #endregion

        #region Memory saving/loading

        public void SaveMemory(string path, NetworkStructure networkStructure)
        {
            // Deleting old memory file:
            FileManager.PrepareToSaveMemory(path, networkStructure);

            // Saving
            int foreachIndex = 0;

            foreach (Layer layer in _layerList)
            {
                layer.SaveMemory(foreachIndex, path);
                foreachIndex++;
            }
        }

        #region Database operations

        public void SaveMemoryToDB(int iterations, string networkStructure, Guid userId, DBInserter dbInserter)
        {
            // Saving network info:
            dbInserter.InsertNetwork(Id, userId, iterations, networkStructure);

            // Saving layers info:
            int foreachIndex = 0;

            foreach (Layer layer in _layerList)
            {
                layer.SaveMemoryToDB(Id, userId, foreachIndex, dbInserter);
                foreachIndex++;
            }
        }

        public void DBMemoryAbort(DBDeleter dbDeleter)
        {
            // Aborting saving of layers:
            foreach (Layer layer in _layerList)
            {
                layer.DBMemoryAbort(Id, dbDeleter);
            }

            // Aborting saving of network:
            dbDeleter.DeleteFromTableNetworks(Id);
        }

        #endregion

        #endregion
    }
}
