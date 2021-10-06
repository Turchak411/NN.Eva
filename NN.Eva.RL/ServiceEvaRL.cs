using System;
using System.Diagnostics;
using NN.Eva.Models;
using NN.Eva.Services;
using NN.Eva.RL.Services;
using NN.Eva.RL.Models;

namespace NN.Eva.RL
{
    public class ServiceEvaRL
    {
        private RLManager _RLManager = null;

        private NetworkStructure _networkStructure = null;

        /// <summary>
        /// Create RL-Agent
        /// </summary>
        /// <param name="memoryFolderName"></param>
        /// <param name="networkStructure"></param>
        /// <returns>Returns success result of Agent creating</returns>
        public bool CreateAgent(string memoryFolderName,
                                NetworkStructure networkStructure)
        {
            _networkStructure = networkStructure;

            if (FileManager.CheckMemoryIntegrity(networkStructure, memoryFolderName))
            {
                try
                {
                    _RLManager = new RLManager(networkStructure);
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
        /// Training tick for RL-Agent
        /// </summary>
        /// <param name="trainingModel"></param>
        /// <param name="processPriorityClass"></param>
        public double[] TrainingTick(RLTrainingModel trainingModel, 
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

            if (_RLManager.CheckMemory(trainingModel.MemoryFolder))
            {
                double[] agentResult = _RLManager.UseAgent(trainingModel, true);

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
    }
}
