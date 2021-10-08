using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Threading;
using NN.Eva.Core;
using NN.Eva.Core.Database;
using NN.Eva.Models;
using NN.Eva.Models.Database;
using NN.Eva.Services;
using NN.Eva.Models.RL;
using NN.Eva.Extensions;

namespace NN.Eva.RL.Services
{
    public class RLManager
    {
        private NeuralNetwork _net;

        private NetworkStructure _networkStructure;

        private TrainingConfigurationLite _trainingConfiguration;

        private RLConfigModel _configModel;

        #region Local services

        private MemoryChecker _memoryChecker;

        private DatasetGenerator _datasetGenerator;

        #endregion

        /// <summary>
        /// Global current iterations
        /// </summary>
        public int IterationsDone { get; private set; } = 0;

        public RLManager(NetworkStructure networkStructure, TrainingConfigurationLite trainingConfiguration, RLConfigModel configModel)
        {
            _networkStructure = networkStructure;
            _trainingConfiguration = trainingConfiguration;
            _configModel = configModel;

            _memoryChecker = new MemoryChecker();
            _datasetGenerator = new DatasetGenerator();

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

        public double[] UseAgent(RLWorkingModel workingModel, bool withTraining = false)
        {
            try 
            {
                if(Enumerable.SequenceEqual(workingModel.CurrentEnvironment, workingModel.FailureEnvironment) &&
                   withTraining)
                {
                    HandleAgentFailure();
                    return null;
                }

                return Handle(workingModel, withTraining);
            }
            catch (Exception ex)
            {
                Logger.LogError(ErrorType.TrainError, ex);
                return new double[_configModel.ActionsCount];
            }
        }

        private double[] Handle(RLWorkingModel workingModel, bool withTraining = false)
        {
            string handlingErrorText = "";
            double[] agentQValues = new double[_configModel.ActionsCount];

            for(int i = 0; i < agentQValues.Length; i++)
            {
                double[] actionsVector = new double[_configModel.ActionsCount];
                actionsVector[i] = 1;

                agentQValues[i] = _net.Handle(workingModel.CurrentEnvironment.ConcatArray(actionsVector), ref handlingErrorText)[0];
            }

            if(handlingErrorText != "")
            {
                Logger.LogError(ErrorType.TrainError, handlingErrorText);
            }

            // Writing to history tail:
            // By-value coping environment array:
            double[] newEnvironment = new double[workingModel.CurrentEnvironment.Length];
            Array.Copy(workingModel.CurrentEnvironment, newEnvironment, workingModel.CurrentEnvironment.Length);

            // User greedy strategy feature:
            if (withTraining)
            {
                double chance = 0.04; // 4%
                agentQValues = UseGreedyStrategy(agentQValues, chance);
            }

            _configModel.MainTail.Add(
                new RLTail
                {
                    Environment = newEnvironment,
                    QValues = agentQValues,
                    ActionIndex = GetMaxIndex(agentQValues)
                });

            if(_configModel.MainTail.Count > _configModel.MainTailMaxLength)
            {
                _configModel.FantomTail.Add(
                    new RLTail
                    {
                        Environment = _configModel.MainTail[0].Environment,
                        QValues = _configModel.MainTail[0].QValues,
                        ActionIndex = _configModel.MainTail[0].ActionIndex
                    });

                _configModel.MainTail.RemoveAt(0);

                if(_configModel.FantomTail.Count > _configModel.FantomTailMaxLength)
                {
                    _configModel.FantomTail.RemoveAt(0);
                }
            }

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

        private double[] UseGreedyStrategy(double[] values, double chance)
        {
            Random rnd = new Random();

            if(rnd.NextDouble() < chance)
            {
                int fakeIndex = rnd.Next(values.Length);
                int maxIndex = GetMaxIndex(values);

                double temp = values[fakeIndex];
                values[fakeIndex] = values[maxIndex];
                values[maxIndex] = temp;

                Console.WriteLine("сработала жадная стратегия");
            }

            return values;
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
                qValues[i] = (qValues[i] - minValue) / (maxValue - minValue + 1);
            }

            return qValues;
        }

        #region Training

        private void HandleAgentFailure()
        {
            // Updating values for tails:
            _configModel.FantomTail = UpdateTail(_configModel.FantomTail, _configModel.PositivePrice);
            _configModel.MainTail = UpdateTail(_configModel.MainTail, _configModel.NegativePrice);

            // Re-training agent:
            RetrainAgent(_trainingConfiguration.EndIteration - _trainingConfiguration.StartIteration, true);

            // Save memory:
            _net.SaveMemory(_trainingConfiguration.MemoryFolder + "//memory.txt", _networkStructure);

            Console.WriteLine("Correcting success for Agent's memory!");
        }

        private List<RLTail> UpdateTail(List<RLTail> tail, double changingValue)
        {
            for(int i = 0; i < tail.Count; i++)
            {
                tail[i].QValues[tail[i].ActionIndex] += changingValue;
            }

            return tail;
        }

        private void RetrainAgent(int iterationsToPause, bool unsafeTrainingMode = false)
        {
            IterationsDone += _trainingConfiguration.EndIteration - _trainingConfiguration.StartIteration;

            // Generate sets for Agent learning:
            List<double[]> inputDataSets = _datasetGenerator.CreateInputSets(_configModel.FantomTail.Concat(_configModel.MainTail).ToList());
            List<double[]> outputDataSets = _datasetGenerator.CreateOutputSets(_configModel.FantomTail.Concat(_configModel.MainTail).ToList());

            Console.WriteLine("Re-training start...");
            try
            {
                List<TrainingConfiguration> trainingConfigs = InitializeTrainingSubConfigs(iterationsToPause);

                // Initialize teachers:
                SingleNetworkTeacher netSubTeacher = new SingleNetworkTeacher
                {
                    Network = _net,
                    NetworkStructure = _networkStructure,
                    TrainingConfiguration = new TrainingConfiguration
                    {
                        TrainingAlgorithmType = TrainingAlgorithmType.BProp,
                        StartIteration = _trainingConfiguration.StartIteration,
                        EndIteration = _trainingConfiguration.EndIteration,
                        MemoryFolder = _trainingConfiguration.MemoryFolder
                    },
                    InputDatasets = inputDataSets.ToArray(),
                    OutputDatasets = outputDataSets.ToArray(),
                    SafeTrainingMode = !unsafeTrainingMode
                };

                // Iteration multithreading train:
                for (int j = 0; j < trainingConfigs.Count; j++)
                {
                    netSubTeacher.TrainingConfiguration = trainingConfigs[j];

                    Thread thread = new Thread(netSubTeacher.Train);
                    thread.Start();
                    Wait(thread);

                    if (!netSubTeacher.LastTrainingSuccess)
                    {
                        return;
                    }

                    if (j != trainingConfigs.Count - 1)
                    {
                        Console.WriteLine("Iterations already finished: " + iterationsToPause * (j + 1));
                    }
                    else
                    {
                        Console.WriteLine("Iterations already finished: " + _trainingConfiguration.EndIteration);
                    }
                }

                // Проведение завершающих операций после обучения модели:
                // В общем случае - получение данных обученной сети от "подучителя":
                _net = netSubTeacher.Network;

                Console.WriteLine("Re-training success!");
            }
            catch (Exception ex)
            {
                Logger.LogError(ErrorType.TrainError, ex);
            }
        }

        private void Wait(Thread thread)
        {
            while (true)
            {
                if (!thread.IsAlive)
                {
                    break;
                }
            }
        }

        private List<TrainingConfiguration> InitializeTrainingSubConfigs(int iterationsToPause)
        {
            List<TrainingConfiguration> trainingConfigs = new List<TrainingConfiguration>();

            int currentIterPosition = _trainingConfiguration.StartIteration;
            while (true)
            {
                if (_trainingConfiguration.EndIteration - currentIterPosition - 1 >= iterationsToPause)
                {
                    var trainingConfigItem = new TrainingConfiguration
                    {
                        TrainingAlgorithmType = TrainingAlgorithmType.BProp,
                        StartIteration = currentIterPosition,
                        EndIteration = currentIterPosition + iterationsToPause,
                        MemoryFolder = _trainingConfiguration.MemoryFolder
                    };

                    trainingConfigs.Add(trainingConfigItem);

                    currentIterPosition += iterationsToPause;
                }
                else
                {
                    var trainingConfigItem = new TrainingConfiguration
                    {
                        TrainingAlgorithmType = TrainingAlgorithmType.BProp,
                        StartIteration = currentIterPosition,
                        EndIteration = _trainingConfiguration.EndIteration,
                        MemoryFolder = _trainingConfiguration.MemoryFolder
                    };

                    trainingConfigs.Add(trainingConfigItem);

                    break;
                }
            }

            Console.WriteLine("Train sub-configuration objects created!");

            return trainingConfigs;
        }

        #endregion

        #region Memory checking

        /// <summary>
        /// Checking network's memory validity
        /// </summary>
        /// <param name="memoryFolder"></param>
        /// <returns></returns>
        public bool CheckMemory()
        {
            bool isValid = true;

            Console.WriteLine("Start memory cheсking...");

            bool isCurrentNetMemoryValid = _networkStructure == null ?
                _memoryChecker.IsFileNotCorrupted(_trainingConfiguration.MemoryFolder + "//memory.txt")
                : _memoryChecker.IsValid(_trainingConfiguration.MemoryFolder + "//memory.txt", _networkStructure) &&
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
            string backupedMemoryFoldersPath = $"{memoryFolder}//{backupsDirectoryName}//{DateTime.Now.Day}_{DateTime.Now.Month}_{DateTime.Now.Year}_{DateTime.Now.Ticks}_{IterationsDone}";

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
                _net.SaveMemoryToDB(IterationsDone, networkStructure, userId, dbInserter);
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