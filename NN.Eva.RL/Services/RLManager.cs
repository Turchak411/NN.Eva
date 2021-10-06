using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NN.Eva.Core;
using NN.Eva.Core.Database;
using NN.Eva.Models;
using NN.Eva.Models.Database;
using NN.Eva.RL.Models;
using NN.Eva.Services;

namespace NN.Eva.RL.Services
{
    public class RLManager
    {
        private NeuralNetwork _net;

        private NetworkStructure _networkStructure;

        /// <summary>
        /// Local services
        /// </summary>
        private MemoryChecker _memoryChecker;

        /// <summary>
        /// Global current iterations
        /// </summary>
        public int Iteration { get; private set; } = 0;

        /// <summary>
        /// Positive price for agent
        /// </summary>
        public double PositivePrice { get; set; }

        /// <summary>
        /// Negative price for agent
        /// </summary>
        public double NegativePrice { get; set; }

        public RLManager(NetworkStructure networkStructure)
        {
            _networkStructure = networkStructure;

            _memoryChecker = new MemoryChecker();

            if (!_memoryChecker.IsValid(FileManager.MemoryFolderPath + "//.clear//memoryClear.txt", networkStructure))
            {
                Logger.LogError(ErrorType.MemoryInitializeError);
                return;
            }

            try
            {
                // Ицициализация сети по одинаковому шаблону:
                _net = new NeuralNetwork(networkStructure.NeuronsByLayers, "memory.txt", networkStructure.Alpha);
            }
            catch (Exception ex)
            {
                Logger.LogError(ErrorType.MemoryInitializeError, ex);
            }
        }

        #region Training

        public double[] UseAgent(RLTrainingModel trainingModel, bool withTraining = false)
        {
            try 
            {
                if(trainingModel.CurrentEnvironment == trainingModel.FailureEnvironment &&
                   withTraining)
                {
                    HandleAgentFailure(trainingModel);
                    return null;
                }

                return Handle(trainingModel);
            }
            catch (Exception ex)
            {
                Logger.LogError(ErrorType.TrainError, ex);
                return new double[_networkStructure.NeuronsByLayers[_networkStructure.NeuronsByLayers.Length - 1]];
            }
        }

        private double[] Handle(RLTrainingModel trainingModel)
        {
            string handlingErrorText = "";
            double[] agentQValues = new double[trainingModel.ActionsCount];

            for(int i = 0; i < agentQValues.Length; i++)
            {
                double[] actionsVector = new double[trainingModel.ActionsCount];
                actionsVector[i] = 1;

                agentQValues[i] = _net.Handle((double[])trainingModel.CurrentEnvironment.Concat(actionsVector), ref handlingErrorText)[0];
            }

            if(handlingErrorText != "")
            {
                Logger.LogError(ErrorType.TrainError, handlingErrorText);
            }

            // Writing to history tail:
            trainingModel.MainTail.Add(
                new RLTail
                {
                    Environment = trainingModel.CurrentEnvironment,
                    QValues = agentQValues,
                    ActionIndex = GetMaxIndex(agentQValues)
                });

            trainingModel.FantomTail.Add(trainingModel.MainTail[0]);

            trainingModel.MainTail.RemoveAt(0);
            trainingModel.FantomTail.RemoveAt(0);

            return GetNormalizedResultVector(agentQValues);
        }

        private int GetMaxIndex(double[] qValues)
        {
            double maxValue = qValues.Max();

            for(int i = 0; i < qValues.Length - 1; i++)
            {
                if(qValues[i] != maxValue)
                {
                    return i;
                }
            }

            return qValues.Length - 1;
        }

        private double[] GetNormalizedResultVector(double[] qValues)
        {
            double minValueOrig = qValues.Min();

            for (int i = 0; i < qValues.Length; i++)
            {
                qValues[i] -= minValueOrig;
            }

            double minValue = qValues.Min();
            double maxValue = qValues.Max();

            for(int i = 0; i < qValues.Length; i++)
            {
                qValues[i] = (qValues[i] - minValue) / (maxValue - minValue);
            }

            return qValues;
        }

        private void HandleAgentFailure(RLTrainingModel trainingModel)
        {
            // TODO:


            _net.SaveMemory(trainingModel.MemoryFullPath, _networkStructure);

            Console.WriteLine("Correcting success for Agent's memory!");
        }

        #endregion

        #region Memory checking

        /// <summary>
        /// Checking network's memory validity
        /// </summary>
        /// <param name="memoryFolder"></param>
        /// <returns></returns>
        public bool CheckMemory(string memoryFolder = "Memory")
        {
            bool isValid = true;

            Console.WriteLine("Start memory cheсking...");

            bool isCurrentNetMemoryValid = _networkStructure == null ?
                _memoryChecker.IsFileNotCorrupted(memoryFolder + "//memory.txt")
                : _memoryChecker.IsValid(memoryFolder + "//memory.txt", _networkStructure) &&
                  FileManager.IsMemoryEqualsDefault("memory.txt");

            if (isCurrentNetMemoryValid)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("memory.txt is valid.");
            }
            else
            {
                isValid = false;
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("memory.txt - is invalid!");
            }

            Console.ForegroundColor = ConsoleColor.Gray;

            return isValid;
        }

        #endregion

        #region Database actions

        /// <summary>
        /// Saving network's memory to local folder OR/AND to database
        /// </summary>
        /// <param name="memoryFolder"></param>
        /// <param name="backupsDirectoryName"></param>
        /// <param name="dbConfig"></param>
        /// <param name="networkStructureInfo"></param>
        public void BackupMemory(string memoryFolder = "Memory", string backupsDirectoryName = ".memory_backups",
            DatabaseConfig dbConfig = null, string networkStructureInfo = "no information")
        {
            // Check for existing main backups-directory:
            if (!Directory.Exists(memoryFolder + "//" + backupsDirectoryName))
            {
                Directory.CreateDirectory(memoryFolder + "//" + backupsDirectoryName);
            }

            // Creating path of backuped memory:
            string backupedMemoryFoldersPath = $"{memoryFolder}//{backupsDirectoryName}//{DateTime.Now.Day}_{DateTime.Now.Month}_{DateTime.Now.Year}_{DateTime.Now.Ticks}_{Iteration}";

            // Check for already-existing sub-directory (trainCount-named):
            if (!Directory.Exists(backupedMemoryFoldersPath))
            {
                Directory.CreateDirectory(backupedMemoryFoldersPath);
            }

            // Saving memory:
            _net.SaveMemory(backupedMemoryFoldersPath + "//memory.txt", _networkStructure);

            // Parsing userID:
            string[] memoryFolderPathArray = memoryFolder.Split('/');
            Guid userId = Guid.Empty;

            for (int i = 0; i < memoryFolderPathArray.Length; i++)
            {
                try
                {
                    userId = Guid.Parse(memoryFolder.Split('/')[i]);
                    break;
                }
                catch { }
            }

            // Saving memory to database:
            if (dbConfig != null)
            {
                Console.WriteLine("Backuping memory to database...");

                SavingMemoryToDB(dbConfig, networkStructureInfo, userId);
            }

            Console.WriteLine("Memory backuped!");
        }

        private void SavingMemoryToDB(DatabaseConfig dbConfig, string networkStructure, Guid userId)
        {
            try
            {
                DBInserter dbInserter = new DBInserter(dbConfig);

                // Saving networks info:
                _net.SaveMemoryToDB(Iteration, networkStructure, userId, dbInserter);
                Console.WriteLine("Networks memory backuped to database successfully!");
            }
            catch (Exception ex)
            {
                Logger.LogError(ErrorType.DBInsertError, "Save memory to database error!\n" + ex);
                Console.WriteLine($" {DateTime.Now} Save memory to database error!\n {ex}");
            }
        }

        /// <summary>
        /// Deleting network's memory from database
        /// </summary>
        /// <param name="dbConfig"></param>
        public void DBMemoryAbort(DatabaseConfig dbConfig)
        {
            try
            {
                DBDeleter dbDeleter = new DBDeleter(dbConfig);

                // Aborting saving network's info:
                _net.DBMemoryAbort(dbDeleter);
                Console.WriteLine("Another network's memory backup aborted successfully!");
            }
            catch (Exception ex)
            {
                Logger.LogError(ErrorType.DBDeleteError, "DB Memory backup aborting error!\n" + ex);
                Console.WriteLine($" {DateTime.Now } DB Memory backup aborting error!\n {ex}");
            }
        }

        /// <summary>
        /// Load network's memory from database
        /// </summary>
        /// <param name="dbConfig"></param>
        /// <param name="networkID"></param>
        /// <param name="destinationMemoryFilePath"></param>
        public void DBMemoryLoad(DatabaseConfig dbConfig, Guid networkID, string destinationMemoryFilePath)
        {
            try
            {
                DBSelector dbSelector = new DBSelector(dbConfig);

                // Creating the general string list of memory:
                List<string> memoryInTextList = new List<string>();

                // Getting all layers ids' ordered by own number in network:
                List<Guid> layersIDList = dbSelector.GetLayerIDList(networkID);

                // As parallel fill the metadata of network:
                NetworkStructure networkStructure = new NetworkStructure();
                networkStructure.NeuronsByLayers = new int[layersIDList.Count];

                for (int i = 0; i < layersIDList.Count; i++)
                {
                    List<Guid> neuronsIDList = dbSelector.GetNeuronsIDList(layersIDList[i]);

                    // Fillin the metadata of count of neurons on current layer:
                    networkStructure.NeuronsByLayers[i] = neuronsIDList.Count;

                    for (int k = 0; k < neuronsIDList.Count; k++)
                    {
                        List<double> weightList = dbSelector.GetWeightsList(neuronsIDList[k]);

                        // Adding memory neuron data to text list:
                        string neuronDataText = $"layer_{i} neuron_{k} ";

                        for (int j = 0; j < weightList.Count; j++)
                        {
                            neuronDataText += weightList[j];
                        }

                        memoryInTextList.Add(neuronDataText);

                        // Fillin metadata of inputVectorLength of network:
                        if (i == 0 && k == 0)
                        {
                            networkStructure.InputVectorLength = weightList.Count;
                        }
                    }
                }

                // Saving memory from textList:
                FileManager.SaveMemoryFromModel(networkStructure, memoryInTextList, destinationMemoryFilePath);
            }
            catch (Exception ex)
            {
                Logger.LogError(ErrorType.DBMemoryLoadError, "DB Memory loading error!\n" + ex);
                Console.WriteLine($" {DateTime.Now } DB Memory loading error!\n {ex}");
            }
        }

        #endregion
    }
}
