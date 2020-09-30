using System;
using System.Diagnostics;
using MySql.Data.MySqlClient;
using NN.Eva.Core;
using NN.Eva.Models;
using NN.Eva.Models.Database;
using NN.Eva.Services;

namespace NN.Eva
{
    public class ServiceEvaNN
    {
        private FileManager _fileManager;
        private NetworksTeacher _networkTeacher;

        /// <summary>
        /// Creating FFNN
        /// </summary>
        /// <param name="memoryFolderName"></param>
        /// <param name="networkStructure"></param>
        /// <param name="netsCountInAssembly"></param>
        /// <param name="testDatasetPath"></param>
        /// <returns>Returns success result of network creating</returns>
        public bool CreateNetwork(string memoryFolderName, NetworkStructure networkStructure,
                                    string testDatasetPath = null)
        {
            _fileManager = new FileManager(networkStructure, memoryFolderName);

            if(_fileManager.IsMemoryLoadCorrect)
            {
                _networkTeacher = new NetworksTeacher(networkStructure, _fileManager);

                if (testDatasetPath != null)
                {
                    _networkTeacher.TestVectors = _fileManager.LoadTestDataset(testDatasetPath);
                }

                return true;
            }
            else
            {
                return false;
            }
        }

        /// <summary>
        /// Training FFNN
        /// </summary>
        /// <param name="trainingConfiguration"></param>
        /// <param name="iterationToPause"></param>
        /// <param name="printLearnStatistic"></param>
        /// <param name="processPriorityClass"></param>
        public void Train(TrainingConfiguration trainingConfiguration, int iterationToPause = 100, bool printLearnStatistic = false, ProcessPriorityClass processPriorityClass = ProcessPriorityClass.Normal)
        {
            trainingConfiguration.MemoryFolder = trainingConfiguration.MemoryFolder == "" ? "Memory" : trainingConfiguration.MemoryFolder;

            // Set the process priority class:
            Process thisProc = Process.GetCurrentProcess();
            thisProc.PriorityClass = processPriorityClass;

            if (_networkTeacher.CheckMemory(trainingConfiguration.MemoryFolder))
            {
                _networkTeacher.TrainNet(trainingConfiguration, iterationToPause);

                if (printLearnStatistic)
                {
                    _networkTeacher.PrintLearnStatistic(trainingConfiguration, true);
                }
            }
            else
            {
                Console.WriteLine("Train failed! Invalid memory!");
            }
        }

        /// <summary>
        /// Handling double-vector data
        /// </summary>
        /// <param name="data"></param>
        /// <returns>Vector-classes</returns>
        public double[] Handle(double[] data)
        {
            try
            {
                return _networkTeacher.Handle(data);
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Backuping memory to db or local folder or to both ones
        /// </summary>
        /// <param name="memoryFolder"></param>
        /// <param name="dbConnection"></param>
        /// <param name="networkStructure"></param>
        /// <returns>State of operation success</returns>
        public bool BackupMemory(string memoryFolder, DatabaseConfig dbConfig = null, string networkStructureInfo = "no information")
        {
            try
            {
                if (_networkTeacher.CheckMemory(memoryFolder))
                {
                    _networkTeacher.BackupMemory(memoryFolder, ".memory_backups", dbConfig);  
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
        /// Aborting memory of current neural network
        /// </summary>
        /// <param name="dbConnection"></param>
        /// <returns>State of operation success</returns>
        public bool DBMemoryAbort(DatabaseConfig dbConfig)
        {
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

        public bool DBMemoryLoad(DatabaseConfig dbConfig, Guid networkID, string destinationMemoryFilePath)
        {
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
