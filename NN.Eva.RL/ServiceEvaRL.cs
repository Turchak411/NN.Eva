using System;
using System.Diagnostics;
using NN.Eva.Models;
using NN.Eva.Services;
using NN.Eva.RL.Services;
using NN.Eva.RL.Models;
using NN.Eva.Models.Database;

namespace NN.Eva.RL
{
    public class ServiceEvaRL
    {
        private RLManager _RLManager = null;

        private NetworkStructure _networkStructure = null;

        /// <summary>
        /// Create RL-Agent
        /// </summary>
        /// <param name="trainingConfiguration">*Only BProp available to agent retraining</param>
        /// <param name="networkStructure">Network's structure</param>
        /// <returns>Returns success result of Agent creating</returns>
        public bool CreateAgent(TrainingConfigurationLite trainingConfiguration,
                                NetworkStructure networkStructure)
        {
            _networkStructure = networkStructure;

            if (FileManager.CheckMemoryIntegrity(networkStructure, trainingConfiguration.MemoryFolder))
            {
                try
                {
                    _RLManager = new RLManager(networkStructure, trainingConfiguration);
                }
                catch
                {
                    return false;
                }

                return true;
            }
            else
            {
                return false;
            }
        }

        /// <summary>
        /// Handling data by Agent
        /// </summary>
        /// <param name="environmentInteractionModel">Environment interaction model</param>
        /// <returns>Vector-classes</returns>
        public double[] UseAgent(RLWorkingModel environmentInteractionModel)
        {
            if (_RLManager == null)
            {
                Logger.LogError(ErrorType.TrainError, "Training failed! Please, create the Network first!");
                return null;
            }

            try
            {
                return _RLManager.UseAgent(environmentInteractionModel);
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Training tick for RL-Agent
        /// </summary>
        /// <param name="workingModel"></param>
        /// <param name="processPriorityClass"></param>
        public double[] TrainingTick(RLWorkingModel workingModel,
                                     ProcessPriorityClass processPriorityClass = ProcessPriorityClass.Normal)
        {
            // Check for unexistent network:
            if (_RLManager == null)
            {
                Logger.LogError(ErrorType.OperationWithNonexistentNetwork, "Training tick failed!");
                throw new Exception("Training tick failed!");
            }

            // Start process timer:
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            // Set the process priority class:
            Process thisProc = Process.GetCurrentProcess();
            thisProc.PriorityClass = processPriorityClass;

            if (_RLManager.CheckMemory())
            {
                double[] agentResult = _RLManager.UseAgent(workingModel, true);

                // Stopping timer and print spend time in [HH:MM:SS]:
                stopWatch.Stop();
                TimeSpan ts = stopWatch.Elapsed;

                string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}", ts.Hours, ts.Minutes, ts.Seconds);
                Console.WriteLine("Time spend: " + elapsedTime);

                return agentResult;
            }
            else
            {
                stopWatch.Stop();
                Console.WriteLine("Training tick failed!");
                throw new Exception("Training tick failed!");
            }
        }

        /// <summary>
        /// Backuping Agent's memory to db OR/AND local folder
        /// </summary>
        /// <param name="memoryFolder"></param>
        /// <param name="dbConfig"></param>
        /// <param name="networkStructureInfo"></param>
        /// <returns>State of operation success</returns>
        public bool BackupMemory(string memoryFolder, DatabaseConfig dbConfig = null, string networkStructureInfo = "no information")
        {
            if (_RLManager == null)
            {
                Logger.LogError(ErrorType.OperationWithNonexistentNetwork, "Database memory backuping failed!");
                return false;
            }

            try
            {
                if (_RLManager.CheckMemory())
                {
                    _RLManager.BackupMemory(memoryFolder, ".memory_backups", dbConfig, networkStructureInfo);
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
        /// Aborting agent's memory from database
        /// </summary>
        /// <param name="dbConfig"></param>
        /// <returns>State of operation success</returns>
        public bool DBMemoryAbort(DatabaseConfig dbConfig)
        {
            if (_RLManager == null)
            {
                Logger.LogError(ErrorType.OperationWithNonexistentNetwork, "Database memory aborting failed!");
                return false;
            }

            try
            {
                _RLManager.DBMemoryAbort(dbConfig);

                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Loading agent's memory from database
        /// </summary>
        /// <param name="dbConfig"></param>
        /// <param name="networkID"></param>
        /// <param name="destinationMemoryFilePath"></param>
        /// <returns></returns>
        public bool DBMemoryLoad(DatabaseConfig dbConfig, Guid networkID, string destinationMemoryFilePath)
        {
            if (_RLManager == null)
            {
                Logger.LogError(ErrorType.OperationWithNonexistentNetwork, "Database memory loading failed!");
                return false;
            }

            try
            {
                _RLManager.DBMemoryLoad(dbConfig, networkID, destinationMemoryFilePath);

                return true;
            }
            catch
            {
                return false;
            }
        }
    }
}
