using NN.Eva.Models;
using System.Collections.Generic;

namespace NN.Eva.Services
{
    public class DatasetChecker
    {
        public bool CheckInputDataset(ref string errorMessage, string datasetFilepath, NetworkStructure networkStructure)
        {
            List<double[]> dataset = FileManager.LoadTrainingDataset(datasetFilepath);

            if (dataset.Count == 0)
            {
                errorMessage += "Input dataset is empty!\n";
                return false;
            }

            for(int i = 0; i < dataset.Count; i++)
            {
                if (dataset[i].Length != networkStructure.InputVectorLength)
                {
                    errorMessage += $"-----------------------------------------\n" +
                                    $"Input dataset's rows does not match existent network's structure!\n" +
                                    $"Network structure requires: { networkStructure.InputVectorLength } rows.\n" +
                                    $"Input dataset has: { dataset[i].Length } rows.\n" +
                                    $"-----------------------------------------\n";
                    return false;
                }
            }

            errorMessage += "";
            return true;
        }

        public bool CheckOutputDataset(ref string errorMessage, string datasetFilepath, NetworkStructure networkStructure)
        {
            List<double[]> dataset = FileManager.LoadTrainingDataset(datasetFilepath);

            if (dataset.Count == 0)
            {
                errorMessage += "Output dataset is empty!\n";
                return false;
            }

            for (int i = 0; i < dataset.Count; i++)
            {
                if (dataset[i].Length != networkStructure.NeuronsByLayers[networkStructure.NeuronsByLayers.Length - 1])
                {
                    errorMessage += $"-----------------------------------------\n" +
                                    $"Output dataset's rows does not match existent network's structure!\n" +
                                    $"Network structure requires: { networkStructure.NeuronsByLayers[networkStructure.NeuronsByLayers.Length - 1] } rows.\n" +
                                    $"Output dataset has: { dataset[i].Length } rows.\n" +
                                    $"-----------------------------------------\n";
                    return false;
                }
            }

            errorMessage += "no_errors";
            return true;
        }

        public bool CheckTrainingSetsCounts(ref string errorMessage, string inputSetFilepath, string outputSetFilepath)
        {
            List<double[]> inputSet = FileManager.LoadTrainingDataset(inputSetFilepath);
            List<double[]> outputSet = FileManager.LoadTrainingDataset(outputSetFilepath);

            if (inputSet.Count != outputSet.Count)
            {
                errorMessage += $"-----------------------------------------\n" +
                                $"Training dataset's rows count does not equals!\n" +
                                $"Output dataset has: { inputSet.Count } rows.\n" +
                                $"Output dataset has: { outputSet.Count } rows.\n" +
                                $"-----------------------------------------\n";
                return false;
            }

            errorMessage += "no_errors";
            return true;
        }
    }
}
