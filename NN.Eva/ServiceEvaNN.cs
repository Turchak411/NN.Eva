using System;
using System.Diagnostics;
using MySql.Data.MySqlClient;
using NN.Eva.Core;
using NN.Eva.Models;
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
        public void CreateNetwork(string memoryFolderName, NetworkStructure networkStructure,
                                  int netsCountInAssembly = 1,
                                  string testDatasetPath = null)
        {
            _fileManager = new FileManager(networkStructure, memoryFolderName);

            _networkTeacher = new NetworksTeacher(networkStructure, netsCountInAssembly, _fileManager);

            if (testDatasetPath != null)
            {
                _networkTeacher.TestVectors = _fileManager.LoadTestDataset(testDatasetPath);
            }
        }

        /// <summary>
        /// Training FFNN
        /// </summary>
        /// <param name="trainingConfiguration"></param>
        /// <param name="iterationToPause"></param>
        /// <param name="printLearnStatistic"></param>
        /// <param name="processPriorityClass"></param>
        public void Train(TrainingConfiguration trainingConfiguration, int iterationToPause = 100, bool printLearnStatistic = false, NetworkStructure netStructure = null, ProcessPriorityClass processPriorityClass = ProcessPriorityClass.Normal)
        {
            trainingConfiguration.MemoryFolder = trainingConfiguration.MemoryFolder == "" ? "Memory" : trainingConfiguration.MemoryFolder;

            // Set the process priority class:
            Process thisProc = Process.GetCurrentProcess();
            thisProc.PriorityClass = ProcessPriorityClass.AboveNormal;

            if (_networkTeacher.CheckMemory(trainingConfiguration.MemoryFolder, netStructure))
            {
                _networkTeacher.TrainNets(trainingConfiguration, iterationToPause);

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
        /// Handling double-vector data as nets assembly
        /// </summary>
        /// <param name="data"></param>
        /// <returns>Vector-classes</returns>
        public double[] HandleAsAssembly(double[] data)
        {
            try
            {
                return _networkTeacher.HandleAsAssembly(data);
            }
            catch
            {
                return null;
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
        public bool BackupMemory(string memoryFolder, MySqlConnection dbConnection, string networkStructure = "no information")
        {
            try
            {
                if (_networkTeacher.CheckMemory(memoryFolder))
                {
                    _networkTeacher.BackupMemory(memoryFolder, ".memory_backups", dbConnection, networkStructure);
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
        public bool DBMemoryAbort(MySqlConnection dbConnection)
        {
            try
            {
                _networkTeacher.DBMemoryAbort(dbConnection);

                return true;
            }
            catch
            {
                return false;
            }
        }
    }
}
