using System.Collections.Generic;
using NN.Eva.Models;

namespace NN.Eva.Services
{
    public class DatasetChecker
    {
        public bool CheckInputDataset(string datasetFilepath, NetworkStructure networkStructure)
        {
            List<double[]> dataset = FileManager.LoadTrainingDataset(datasetFilepath);

            if (dataset.Count == 0)
                return false;

            for(int i = 0; i < dataset.Count; i++)
            {
                if (dataset[i].Length != networkStructure.InputVectorLength)
                    return false;
            }

            return true;
        }

        public bool CheckOutputDataset(string datasetFilepath, NetworkStructure networkStructure)
        {
            List<double[]> dataset = FileManager.LoadTrainingDataset(datasetFilepath);

            if (dataset.Count == 0)
                return false;

            for (int i = 0; i < dataset.Count; i++)
            {
                if (dataset[i].Length != networkStructure.NeuronsByLayers[networkStructure.NeuronsByLayers.Length - 1])
                    return false;
            }

            return true;
        }
    }
}
