using System;
using System.Collections.Generic;
using System.Text;

namespace NN.Eva.Core
{
    public class NeuralNetwork
    {
        public Guid Id { get; set; } = Guid.NewGuid();

        protected List<Layer> _layerList = new List<Layer>();
        private readonly FileManager _fileManager;

        protected NeuralNetwork() { }

        public NeuralNetwork(int receptorsNumber, int[] neuronsNumberByLayers, FileManager fileManager)
        {
            _fileManager = fileManager;

            Layer firstLayer = new Layer(neuronsNumberByLayers[0], receptorsNumber, 0, fileManager);
            _layerList.Add(firstLayer);

            for (int i = 1; i < neuronsNumberByLayers.Length; i++)
            {
                Layer layer = new Layer(neuronsNumberByLayers[i], neuronsNumberByLayers[i - 1], i, fileManager);
                _layerList.Add(layer);
            }
        }

        public NeuralNetwork(int receptorsNumber, int[] neuronsNumberByLayers, FileManager fileManager, string memoryPath)
        {
            _fileManager = fileManager;

            Layer firstLayer = new Layer(neuronsNumberByLayers[0], receptorsNumber, 0, fileManager, memoryPath);
            _layerList.Add(firstLayer);

            for (int i = 1; i < neuronsNumberByLayers.Length; i++)
            {
                Layer layer = new Layer(neuronsNumberByLayers[i], neuronsNumberByLayers[i - 1], i, fileManager, memoryPath);
                _layerList.Add(layer);
            }
        }

        public double[] Handle(double[] data)
        {
            double[] tempData = data;

            for (int i = 0; i < _layerList.Count; i++)
            {
                tempData = _layerList[i].Handle(tempData);
            }

            // There is one double value at the last handle

            return HandleNetAnwser(tempData);
        }

        private double[] HandleNetAnwser(double[] netResult)
        {
            return netResult;
        }

        public void Teach(double[] data, double[] rightAnwsersSet, double learnSpeed)
        {
            // Подсчет ошибки:
            _layerList[_layerList.Count - 1].CalcErrorAsOut(rightAnwsersSet);

            for (int i = _layerList.Count - 2; i >= 0; i--)
            {
                double[][] nextLayerWeights = _layerList[i + 1].GetWeights();
                double[] nextLayerErrors = _layerList[i + 1].GetErrors();

                _layerList[i].CalcErrorAsHidden(nextLayerWeights, nextLayerErrors);
            }

            // Корректировка весов нейронов:
            double[] anwsersFromPrewLayer = data;

            for (int i = 0; i < _layerList.Count; i++)
            {
                _layerList[i].ChangeWeights(learnSpeed, anwsersFromPrewLayer);
                anwsersFromPrewLayer = _layerList[i].GetLastAnwsers();
            }
        }

        public void SaveMemory()
        {
            // Deleting old memory file:
            _fileManager.PrepareToSaveMemory();

            // Saving
            for (int i = 0; i < _layerList.Count; i++)
            {
                _layerList[i].SaveMemory(_fileManager, i);
            }
        }

        public void SaveMemory(string path)
        {
            // Deleting old memory file:
            _fileManager.PrepareToSaveMemory(path);

            // Saving
            for (int i = 0; i < _layerList.Count; i++)
            {
                _layerList[i].SaveMemory(_fileManager, i, path);
            }
        }

        public void SaveMemoryToDB(int iterations, string networkStructure, Guid userId, DBInserter dbInserter)
        {
            // Saving network info:
            dbInserter.InsertNetwork(Id, userId, iterations, networkStructure);

            // Saving layers info:
            for (int i = 0; i < _layerList.Count; i++)
            {
                _layerList[i].SaveMemoryToDB(Id, i, dbInserter);
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
    }
}
