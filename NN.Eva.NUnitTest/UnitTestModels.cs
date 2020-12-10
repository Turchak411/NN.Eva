using NN.Eva.Models;
using NUnit.Framework;

namespace NN.Eva.NUnitTest
{
    public class UnitTestModels
    {
        [Test]
        public void TestNetworkStructureModel()
        {
            // Arrange:
            NetworkStructure networkStructure = new NetworkStructure
            {
                InputVectorLength = 1,
                NeuronsByLayers = new int[] { 1 }
            };

            // Act & Assert:
            Assert.IsNotNull(networkStructure);
        }

        [Test]
        public void TestTrainingConfiguration()
        {
            // Arrange:
            TrainingConfiguration trainingConfig = new TrainingConfiguration
            {
                StartIteration = 0,
                EndIteration = 1,
                InputDatasetFilename = "test",
                OutputDatasetFilename = "test",
                MemoryFolder = "test",
                TrainingAlgorithmType = TrainingAlgorithmType.BProp
            };

            // Act & Assert:
            Assert.IsNotNull(trainingConfig);
        }

        [Test]
        public void TestTrainObject()
        {
            // Arrange:
            TrainObject trainObject = new TrainObject("test", new double[] { 0.0, 0.0 });

            // Act & Assert:
            Assert.IsNotNull(trainObject);
        }
    }
}
