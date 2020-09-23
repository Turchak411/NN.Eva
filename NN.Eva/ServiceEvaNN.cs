using System;
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
                                  string testDatasetsPath = null)
        {
            //#region Set process settings

            //Process thisProc = Process.GetCurrentProcess();
            //thisProc.PriorityClass = ProcessPriorityClass.AboveNormal;

            //#endregion

            _fileManager = new FileManager(networkStructure);

            _networkTeacher = new NetworksTeacher(networkStructure, netsCountInAssembly, _fileManager, memoryFolderName);

            if (testDatasetsPath != null)
            {
                _networkTeacher.TestVectors = _fileManager.LoadDatasets(testDatasetsPath);
            }
        }

        public void Train(TrainConfiguration trainConfiguration, int iterationToPause = 100)
        {
            if (_networkTeacher.CheckMemory(trainConfiguration.MemoryFolder))
            {
                _networkTeacher.TrainNets(trainConfiguration, iterationToPause);

                //_networkTeacher.PrintLearnStatistic(trainConfiguration, true);
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

        public void DBMemoryAbort(MySqlConnection dbConnection)
        {
            _networkTeacher.DBMemoryAbort(dbConnection);
        }
    }
}
