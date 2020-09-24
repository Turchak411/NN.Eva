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

        public void Train(TrainingConfiguration trainingConfiguration, int iterationToPause = 100, bool printLearnStatistic = false, ProcessPriorityClass processPriorityClass = ProcessPriorityClass.Normal)
        {
            trainingConfiguration.MemoryFolder = trainingConfiguration.MemoryFolder == "" ? "Memory" : trainingConfiguration.MemoryFolder;

            // Set the process priority class:
            Process thisProc = Process.GetCurrentProcess();
            thisProc.PriorityClass = ProcessPriorityClass.AboveNormal;

            if (_networkTeacher.CheckMemory(trainingConfiguration.MemoryFolder))
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

        public void DBMemoryAbort(MySqlConnection dbConnection)
        {
            _networkTeacher.DBMemoryAbort(dbConnection);
        }
    }
}
