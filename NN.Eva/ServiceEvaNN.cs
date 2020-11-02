using System;
using System.Diagnostics;
using NN.Eva.Core;
using NN.Eva.Models;
using NN.Eva.Models.Database;
using NN.Eva.Services;

namespace NN.Eva
{
    public class ServiceEvaNN
    {
        private NetworksTeacher _networkTeacher = null;

        private NetworkStructure _networkStructure = null;

        /// <summary>
        /// Creating FeedForward - Neural Network
        /// </summary>
        /// <param name="memoryFolderName"></param>
        /// <param name="networkStructure"></param>
        /// <param name="testDatasetPath"></param>
        /// <returns>Returns success result of network creating</returns>
        public bool CreateNetwork(string memoryFolderName,
                                  NetworkStructure networkStructure,
                                  string testDatasetPath = null)
        {
            _networkStructure = networkStructure;

            if (FileManager.CheckMemoryIntegrity(networkStructure, memoryFolderName))
            {
                try
                {
                    _networkTeacher = new NetworksTeacher(networkStructure);
                }
                catch
                {
                    return false;
                }

                if (testDatasetPath != null)
                {
                    _networkTeacher.TestVectors = FileManager.LoadTestDataset(testDatasetPath);
                }

                return true;
            }
            else
            {
                return false;
            }
        }

        /// <summary>
        /// Training FeedForward - NeuralNetwork
        /// </summary>
        /// <param name="trainingConfiguration"></param>
        /// <param name="printLearnStatistic"></param>
        /// <param name="processPriorityClass"></param>
        /// <param name="unsafeTrainingMode"></param>
        /// <param name="iterationsToPause"></param>
        public void Train(TrainingConfiguration trainingConfiguration, 
                          bool printLearnStatistic = false,
                          ProcessPriorityClass processPriorityClass = ProcessPriorityClass.Normal,
                          bool unsafeTrainingMode = false,
                          int iterationsToPause = -1)
        {
            // Check for unexistent network:
            if (_networkTeacher == null)
            {
                Logger.LogError(ErrorType.OperationWithNonexistentNetwork, "Training failed!");
                return;
            }

            // Check for set of iterations to pause:
            if (iterationsToPause == -1)
            {
                iterationsToPause = trainingConfiguration.EndIteration - trainingConfiguration.StartIteration;
            }

            // Print warning about using unsafe mode:
            if (unsafeTrainingMode)
            {
                Logger.LogWarning(WarningType.UsingUnsafeTrainingMode);
            }

            trainingConfiguration.MemoryFolder = trainingConfiguration.MemoryFolder == "" ? "Memory" : trainingConfiguration.MemoryFolder;

            // Start process timer:
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            // Set the process priority class:
            Process thisProc = Process.GetCurrentProcess();
            thisProc.PriorityClass = processPriorityClass;

            if (_networkTeacher.CheckMemory(trainingConfiguration.MemoryFolder) && _networkTeacher.CheckDatasets(trainingConfiguration.InputDatasetFilename, trainingConfiguration.OutputDatasetFilename, _networkStructure))
            {
                _networkTeacher.TrainNet(trainingConfiguration, iterationsToPause, unsafeTrainingMode);

                // Stopping timer and print spend time in [HH:MM:SS]:
                stopWatch.Stop();
                TimeSpan ts = stopWatch.Elapsed;

                string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}", ts.Hours, ts.Minutes, ts.Seconds);
                Console.WriteLine("Time spend: " + elapsedTime);

                if (printLearnStatistic)
                {
                    _networkTeacher.PrintLearningStatistic(trainingConfiguration, true, elapsedTime);
                }
            }
            else
            {
                stopWatch.Stop();
                Console.WriteLine("Training failed!");
            }
        }

        /// <summary>
        /// Handling double-vector data
        /// </summary>
        /// <param name="data"></param>
        /// <returns>Vector-classes</returns>
        public double[] Handle(double[] data)
        {
            if (_networkTeacher == null)
            {
                Logger.LogError(ErrorType.TrainError, "Training failed! Please, create the Network first!");
                return null;
            }

            try
            {
                return _networkTeacher.Handle(data);
            }
            catch
            {
                return null;
            }
        }

        public void CalculateStatistic(TrainingConfiguration trainingConfig)
        {
            if (_networkTeacher == null)
            {
                Logger.LogError(ErrorType.OperationWithNonexistentNetwork, "Calculate statistic failed!");
                return;
            }

            _networkTeacher.PrintLearningStatistic(trainingConfig, true);
        }

        /// <summary>
        /// Backuping network's memory to db OR/AND local folder
        /// </summary>
        /// <param name="memoryFolder"></param>
        /// <param name="dbConfig"></param>
        /// <param name="networkStructureInfo"></param>
        /// <returns>State of operation success</returns>
        public bool BackupMemory(string memoryFolder, DatabaseConfig dbConfig = null, string networkStructureInfo = "no information")
        {
            if (_networkTeacher == null)
            {
                Logger.LogError(ErrorType.OperationWithNonexistentNetwork, "Database memory backuping failed!");
                return false;
            }

            try
            {
                if (_networkTeacher.CheckMemory(memoryFolder))
                {
                    _networkTeacher.BackupMemory(memoryFolder, ".memory_backups", dbConfig, networkStructureInfo);  
                }
                else
                {
                    return false;
                }

                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Aborting network's memory from database
        /// </summary>
        /// <param name="dbConfig"></param>
        /// <returns>State of operation success</returns>
        public bool DBMemoryAbort(DatabaseConfig dbConfig)
        {
            if (_networkTeacher == null)
            {
                Logger.LogError(ErrorType.OperationWithNonexistentNetwork, "Database memory aborting failed!");
                return false;
            }

            try
            {
                _networkTeacher.DBMemoryAbort(dbConfig);

                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Loading network's memory from database
        /// </summary>
        /// <param name="dbConfig"></param>
        /// <param name="networkID"></param>
        /// <param name="destinationMemoryFilePath"></param>
        /// <returns></returns>
        public bool DBMemoryLoad(DatabaseConfig dbConfig, Guid networkID, string destinationMemoryFilePath)
        {
            if (_networkTeacher == null)
            {
                Logger.LogError(ErrorType.OperationWithNonexistentNetwork, "Database memory loading failed!");
                return false;
            }

            try
            {
                _networkTeacher.DBMemoryLoad(dbConfig, networkID, destinationMemoryFilePath);

                return true;
            }
            catch
            {
                return false;
            }
        }
    }
}
