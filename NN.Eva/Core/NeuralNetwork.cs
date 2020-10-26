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

        protected List<Layer> _layerList = new List<Layer>();

        private readonly FileManager _fileManager;

        protected NeuralNetwork() { }

        public NeuralNetwork(int[] neuronsNumberByLayers, FileManager fileManager)
        {
            _fileManager = fileManager;

            Layer firstLayer = new Layer(neuronsNumberByLayers[0], 0, fileManager);
            _layerList.Add(firstLayer);

            for (int i = 1; i < neuronsNumberByLayers.Length; i++)
            {
                Layer layer = new Layer(neuronsNumberByLayers[i], i, fileManager);
                _layerList.Add(layer);
            }
        }

        public NeuralNetwork(int[] neuronsNumberByLayers, FileManager fileManager, string memoryPath)
        {
            _fileManager = fileManager;

            Layer firstLayer = new Layer(neuronsNumberByLayers[0], 0, fileManager, memoryPath);
            _layerList.Add(firstLayer);

            for (int i = 1; i < neuronsNumberByLayers.Length; i++)
            {
                Layer layer = new Layer(neuronsNumberByLayers[i], i, fileManager, memoryPath);
                _layerList.Add(layer);
            }
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

            for (int i = 0; i < _layerList.Count; i++)
            {
                tempData = _layerList[i].Handle(tempData);
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

            for (int i = 0; i < _layerList.Count; i++)
            {
                tempData = _layerList[i].Handle(tempData);
            }

            // There is one double value at the last handle
            return tempData;
        }

        #endregion

        #region Teaching

        public void CalculateErrors(double[] rightAnswersSet)
        {
            _layerList[_layerList.Count - 1].CalcErrorAsOut(rightAnswersSet);

            for (int i = _layerList.Count - 2; i >= 0; i--)
            {
                double[][] nextLayerWeights = _layerList[i + 1].GetWeights();
                double[] nextLayerErrors = _layerList[i + 1].GetNeuronErrors();

                _layerList[i].CalcErrorAsHidden(nextLayerWeights, nextLayerErrors);
            }
        }

        public List<double[]> GetNeuronErrors()
        {
            List<double[]> errorList = new List<double[]>();

            for (int i = 0; i < _layerList.Count; i++)
            {
                errorList.Add(_layerList[i].GetNeuronErrors());
            }

            return errorList;
        }

        public List<double[]> GetLastNeuronAnswers()
        {
            List<double[]> answersList = new List<double[]>();

            for (int i = 0; i < _layerList.Count; i++)
            {
                answersList.Add(_layerList[i].GetLastAnswers());
            }

            return answersList;
        }

        public void TeachBProp(double[] data, double[] rightAnswersSet, double learnSpeed)
        {
            // Подсчет ошибки (внутреннее изменение):
            CalculateErrors(rightAnswersSet);

            // Корректировка весов нейронов:
            double[] answersFromPrevLayer = data;

            for (int i = 0; i < _layerList.Count; i++)
            {
                _layerList[i].ChangeWeightsBProp(learnSpeed, answersFromPrevLayer);
                answersFromPrevLayer = _layerList[i].GetLastAnswers();
            }
        }

        public void TeachRProp(List<double[]> updateValues)
        {
            for (int i = 0; i < _layerList.Count; i++)
            {
                _layerList[i].ChangeWeightsRProp(updateValues[i]);
            }
        }

        #endregion

        #region Memory saving/loading

        public void SaveMemory(string path, NetworkStructure networkStructure)
        {
            // Deleting old memory file:
            _fileManager.PrepareToSaveMemory(path, networkStructure);

            // Saving
            for (int i = 0; i < _layerList.Count; i++)
            {
                _layerList[i].SaveMemory(_fileManager, i, path);
            }
        }

        #region Database operations

        public void SaveMemoryToDB(int iterations, string networkStructure, Guid userId, DBInserter dbInserter)
        {
            // Saving network info:
            dbInserter.InsertNetwork(Id, userId, iterations, networkStructure);

            // Saving layers info:
            for (int i = 0; i < _layerList.Count; i++)
            {
                _layerList[i].SaveMemoryToDB(Id, userId, i, dbInserter);
            }
        }

        public void DBMemoryAbort(DBDeleter dbDeleter)
        {
            // Aborting saving of layers:
            for (int i = 0; i < _layerList.Count; i++)
            {
                _layerList[i].DBMemoryAbort(Id, dbDeleter);
            }

            // Aborting saving of network:
            dbDeleter.DeleteFromTableNetworks(Id);
        }

        #endregion

        #endregion
    }
}
