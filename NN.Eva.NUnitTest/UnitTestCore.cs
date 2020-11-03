using System.Diagnostics;
using System.IO;
using NN.Eva.Models;
using NUnit.Framework;

namespace NN.Eva.NUnitTest
{
    public class UnitTestCore
    {
        [SetUp]
        public void Setup() { }

        [Test]
        public void TestBProp()
        {
            // Arrange:
            ServiceEvaNN serviceEvaNN = new ServiceEvaNN();

            NetworkStructure netStructure = new NetworkStructure
            {
                InputVectorLength = 2,
                NeuronsByLayers = new[] { 2, 2, 1 }
            };

            TrainingConfiguration trainConfig = new TrainingConfiguration
            {
                TrainingAlgorithmType = TrainingAlgorithmType.BProp,
                StartIteration = 0,
                EndIteration = 800000,
                InputDatasetFilename = "TrainingSetsTest//inputSets.txt",
                OutputDatasetFilename = "TrainingSetsTest//outputSets.txt",
                MemoryFolder = "MemoryBProp"
            };

            if (!Directory.Exists("TrainingSetsTest"))
            {
                Directory.CreateDirectory("TrainingSetsTest");
                CreateDataSets(trainConfig);
            }

            if (File.Exists(trainConfig.MemoryFolder + "//memory.txt"))
            {
                File.Delete(trainConfig.MemoryFolder + "//memory.txt");
            }

            bool creatingSucceed = serviceEvaNN.CreateNetwork(trainConfig.MemoryFolder, netStructure);

            if (creatingSucceed)
            {
                serviceEvaNN.Train(trainConfig,
                    true,
                    ProcessPriorityClass.Normal,
                    true);
            }

            // Asserts:
            double tolerance = 0.3;

            double[] result0 = serviceEvaNN.Handle(new double[] { 0, 0 });
            Assert.IsNotNull(result0);
            Assert.AreEqual(0.0, result0[0], tolerance);

            double[] result1 = serviceEvaNN.Handle(new double[] { 0, 1 });
            Assert.IsNotNull(result1);
            Assert.AreEqual(0.0, result1[0], tolerance);

            double[] result2 = serviceEvaNN.Handle(new double[] { 1, 0 });
            Assert.IsNotNull(result2);
            Assert.AreEqual(0.0, result2[0], tolerance);

            double[] result3 = serviceEvaNN.Handle(new double[] { 1, 1 });
            Assert.IsNotNull(result3);
            Assert.AreEqual(1.0, result3[0], tolerance);
        }

        [Test]
        public void TestRProp()
        {
            // Arrange:
            ServiceEvaNN serviceEvaNN = new ServiceEvaNN();

            NetworkStructure netStructure = new NetworkStructure
            {
                InputVectorLength = 2,
                NeuronsByLayers = new[] { 23, 15, 1 }
            };

            TrainingConfiguration trainConfig = new TrainingConfiguration
            {
                TrainingAlgorithmType = TrainingAlgorithmType.RProp,
                StartIteration = 0,
                EndIteration = 100,
                InputDatasetFilename = "TrainingSetsTest//inputSets.txt",
                OutputDatasetFilename = "TrainingSetsTest//outputSets.txt",
                MemoryFolder = "MemoryRProp"
            };

            if (!Directory.Exists("TrainingSetsTest"))
            {
                Directory.CreateDirectory("TrainingSetsTest");
                CreateDataSets(trainConfig);
            }

            if (File.Exists(trainConfig.MemoryFolder + "//memory.txt"))
            {
                File.Delete(trainConfig.MemoryFolder + "//memory.txt");
            }

            bool creatingSucceed = serviceEvaNN.CreateNetwork(trainConfig.MemoryFolder, netStructure);

            if (creatingSucceed)
            {
                serviceEvaNN.Train(trainConfig,
                    true,
                    ProcessPriorityClass.Normal,
                    true);
            }

            // Asserts:
            double tolerance = 0.15;

            double[] result0 = serviceEvaNN.Handle(new double[] { 0, 0 });
            Assert.IsNotNull(result0);
            Assert.AreEqual(0.0, result0[0], tolerance);

            double[] result1 = serviceEvaNN.Handle(new double[] { 0, 1 });
            Assert.IsNotNull(result1);
            Assert.AreEqual(0.0, result1[0], tolerance);

            double[] result2 = serviceEvaNN.Handle(new double[] { 1, 0 });
            Assert.IsNotNull(result2);
            Assert.AreEqual(0.0, result2[0], tolerance);

            double[] result3 = serviceEvaNN.Handle(new double[] { 1, 1 });
            Assert.IsNotNull(result3);
            Assert.AreEqual(1.0, result3[0], tolerance);
        }

        private void CreateDataSets(TrainingConfiguration trainingConfiguration)
        {
            // Creating input sets:
            using (StreamWriter fileWriter = new StreamWriter(trainingConfiguration.InputDatasetFilename))
            {
                fileWriter.WriteLine("0 0");
                fileWriter.WriteLine("0 1");
                fileWriter.WriteLine("1 0");
                fileWriter.Write("1 1");
            }

            // Creating input sets:
            using (StreamWriter fileWriter = new StreamWriter(trainingConfiguration.OutputDatasetFilename))
            {
                fileWriter.WriteLine("0");
                fileWriter.WriteLine("0");
                fileWriter.WriteLine("0");
                fileWriter.Write("1");
            }
        }
    }
}