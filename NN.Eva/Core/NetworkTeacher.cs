using NN.Eva.Models;
using NN.Eva.Services;
using System;
using System.IO;
using System.Collections.Generic;
using System.Threading;
using NN.Eva.Core.Database;
using NN.Eva.Models.Database;

namespace NN.Eva.Core
{
    public class NetworksTeacher
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
        public int Iteration { get; set; } = 0;

        /// <summary>
        /// Testing objects
        /// </summary>
        public List<TrainObject> TestVectors { get; set; }

        public NetworksTeacher(NetworkStructure networkStructure)
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
                _net = new NeuralNetwork(networkStructure.NeuronsByLayers, "memory.txt");
            }
            catch (Exception ex)
            {
                Logger.LogError(ErrorType.MemoryInitializeError, ex);
            }
        }

        #region Trained testing

        /// <summary>
        /// Common test data from TestVector datafile
        /// </summary>
        /// <param name="isColorized"></param>
        public void CommonTest(bool isColorized = false)
        {
            if (TestVectors == null) return;

            for (int k = 0; k < TestVectors.Count; k++)
            {
                // Получение ответа:
                string handlingErrorText = "";
                var outputVector = _net.Handle(TestVectors[k]._vectorValues, ref handlingErrorText);

                if (outputVector != null)
                {
                    Console.ForegroundColor = GetColorByActivation(outputVector[0]);
                    Console.Write($"{outputVector[0]:f5}\t");
                }
                else
                {
                    Logger.LogError(ErrorType.NonEqualsInputLengths, handlingErrorText);
                    break;
                }
            }

            Console.ForegroundColor = ConsoleColor.Gray;
            Console.Write('\n');
        }

        private ConsoleColor GetColorByActivation(double value)
        {
            if (value > 0.95)
            {
                return ConsoleColor.Red;
            }

            if (value > 0.8)
            {
                return ConsoleColor.Magenta;
            }

            if (value > 0.5)
            {
                return ConsoleColor.Yellow;
            }

            return ConsoleColor.Gray;
        }

        /// <summary>
        /// Printing network's learning statistic
        /// </summary>
        /// <param name="trainingConfig"></param>
        /// <param name="withLogging"></param>
        public void PrintLearningStatistic(TrainingConfiguration trainingConfig, bool withLogging = false, string elapsedTime = "")
        {
            Console.WriteLine("Start calculating statistic...");

            // Loading memory:
            try
            {
                _net = new NeuralNetwork(_networkStructure.NeuronsByLayers, "memory.txt");
            }
            catch (Exception ex)
            {
                Logger.LogError(ErrorType.MemoryInitializeError, ex);
            }

            // Testing:
            int testPassed = 0;
            int testFailed = 0;

            #region Load data from file

            List<double[]> inputDataSets;
            List<double[]> outputDataSets;

            try
            {
                inputDataSets = FileManager.LoadTrainingDataset(trainingConfig.InputDatasetFilename);
                outputDataSets = FileManager.LoadTrainingDataset(trainingConfig.OutputDatasetFilename);
            }
            catch (Exception ex)
            {
                Logger.LogError(ErrorType.SetMissing, ex);
                return;
            }

            #endregion

            for (int i = 0; i < inputDataSets.Count; i++)
            {
                // Получение ответа:
                string handlingErrorText = "";
                double[] netResult = _net.Handle(inputDataSets[i], ref handlingErrorText);

                if (netResult != null)
                {
                    if (IsVectorsRoughlyEquals(outputDataSets[i], netResult, 0.3))
                    {
                        testPassed++;
                    }
                    else
                    {
                        testFailed++;
                    }
                }
                else
                {
                    Logger.LogError(ErrorType.NonEqualsInputLengths, handlingErrorText);
                    return;
                }
            }

            // Logging (optional):
            if (withLogging)
            {
                Logger.LogTrainingResults(testPassed, testFailed, trainingConfig, elapsedTime);
            }

            Console.WriteLine("Test passed: {0}\nTest failed: {1}\nPercent learned: {2:f2}", testPassed,
                                                                                             testFailed,
                                                                                             (double)testPassed * 100 / (testPassed + testFailed));
        }

        private bool IsVectorsRoughlyEquals(double[] sourceVector0, double[] controlVector1, double equalsPercent)
        {
            // Возвращение неравенства, если длины векторов не совпадают
            if(sourceVector0.Length != controlVector1.Length)
            {
                return false;
            }

            for(int i = 0; i < sourceVector0.Length; i++)
            {
                if (controlVector1[i] < sourceVector0[i] - equalsPercent || controlVector1[i] > sourceVector0[i] + equalsPercent)
                {
                    return false;
                }
            }

            return true;
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

        #region Datasets checking

        public bool CheckDatasets(string inputDatasetFilename, string outputDatasetFilename, NetworkStructure networkStructure)
        {
            Console.WriteLine("Start datasets cheсking...");

            DatasetChecker datasetChecker = new DatasetChecker();

            return CheckSingleDataset(inputDatasetFilename, datasetChecker, networkStructure, true) &&
                   CheckSingleDataset(outputDatasetFilename, datasetChecker, networkStructure, false);
        }

        public bool CheckSingleDataset(string datasetFilename, DatasetChecker datasetChecker, NetworkStructure networkStructure, bool isItInputDataset)
        {
            bool isValid = true;

            Console.WriteLine($"Start dataset \"{datasetFilename}\" cheсking...");

            bool isCurrentDatasetValid = isItInputDataset ? 
                                         datasetChecker.CheckInputDataset(datasetFilename, networkStructure) :
                                         datasetChecker.CheckOutputDataset(datasetFilename, networkStructure);

            if (isCurrentDatasetValid)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"\"{datasetFilename}\" is valid.");
            }
            else
            {
                isValid = false;
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\"{datasetFilename}\" - is invalid!");
            }

            Console.ForegroundColor = ConsoleColor.Gray;

            return isValid;
        }

        #endregion

        #region Training

        /// <summary>
        /// Обучение сети
        /// </summary>
        /// <param name="trainingConfig"></param>
        /// <param name="iterationsToPause"></param>
        /// <param name="unsafeTrainingMode"></param>
        public void TrainNet(TrainingConfiguration trainingConfig, int iterationsToPause, bool unsafeTrainingMode = false)
        {
            Iteration = trainingConfig.EndIteration;

            #region Load data from file

            List<double[]> inputDataSets;
            List<double[]> outputDataSets;

            try
            {
                inputDataSets = FileManager.LoadTrainingDataset(trainingConfig.InputDatasetFilename);
                outputDataSets = FileManager.LoadTrainingDataset(trainingConfig.OutputDatasetFilename);
            }
            catch (Exception ex)
            {
                Logger.LogError(ErrorType.SetMissing, ex);
                return;
            }

            #endregion

            Console.WriteLine("Training start...");
            try
            {
                List<TrainingConfiguration> trainingConfigs = InitializeTrainingSubConfigs(trainingConfig, iterationsToPause);

                // Initialize teachers:
                SingleNetworkTeacher netSubTeacher = new SingleNetworkTeacher
                {
                    Network = _net,
                    NetworkStructure = _networkStructure,
                    TrainingConfiguration = trainingConfig,
                    InputDatasets = inputDataSets,
                    OutputDatasets = outputDataSets,
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
                        Console.WriteLine("Iterations already finished: " + trainingConfig.EndIteration);
                    }

                    // Test after this iteration's part
                    CommonTest(true);
                }

                // Проведение завершающих операций после обучения модели:
                switch (trainingConfig.TrainingAlgorithmType)
                {
                    // В случае обучения по генетическому алгоритму - загрузка памяти из файла:
                    case TrainingAlgorithmType.GeneticAlg:
                    case TrainingAlgorithmType.RProp:
                        // Загрузка только что сохраненной памяти:
                        _net = new NeuralNetwork(_networkStructure.NeuronsByLayers, "memory.txt");
                        break;
                    // В общем случае - получение данных обученной сети от "подучителя":
                    case TrainingAlgorithmType.BProp:
                    default:
                        _net = netSubTeacher.Network;
                        break;
                }

                Console.WriteLine("Training success!");
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
                if(!thread.IsAlive)
                {
                    break;
                }
            }
        }

        private List<TrainingConfiguration> InitializeTrainingSubConfigs(TrainingConfiguration trainingConfig, int iterationsToPause)
        {
            List<TrainingConfiguration> trainingConfigs = new List<TrainingConfiguration>();

            int currentIterPosition = trainingConfig.StartIteration;
            while (true)
            {
                if (trainingConfig.EndIteration - currentIterPosition - 1 >= iterationsToPause)
                {
                    var trainingConfigItem = new TrainingConfiguration
                    {
                        TrainingAlgorithmType = trainingConfig.TrainingAlgorithmType,
                        StartIteration = currentIterPosition,
                        EndIteration = currentIterPosition + iterationsToPause,
                        MemoryFolder = trainingConfig.MemoryFolder,
                        InputDatasetFilename = trainingConfig.InputDatasetFilename,
                        OutputDatasetFilename = trainingConfig.OutputDatasetFilename
                    };

                    trainingConfigs.Add(trainingConfigItem);

                    currentIterPosition += iterationsToPause;
                }
                else
                {
                    var trainingConfigItem = new TrainingConfiguration
                    {
                        TrainingAlgorithmType = trainingConfig.TrainingAlgorithmType,
                        StartIteration = currentIterPosition,
                        EndIteration = trainingConfig.EndIteration,
                        MemoryFolder = trainingConfig.MemoryFolder,
                        InputDatasetFilename = trainingConfig.InputDatasetFilename,
                        OutputDatasetFilename = trainingConfig.OutputDatasetFilename
                    };

                    trainingConfigs.Add(trainingConfigItem);

                    break;
                }
            }

            Console.WriteLine("Train sub-configuration objects created!");

            return trainingConfigs;
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
            Guid userId;

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

                for(int i = 0; i < layersIDList.Count; i++)
                {
                    List<Guid> neuronsIDList = dbSelector.GetNeuronsIDList(layersIDList[i]);

                    // Fillin the metadata of count of neurons on current layer:
                    networkStructure.NeuronsByLayers[i] = neuronsIDList.Count;

                    for(int k = 0; k < neuronsIDList.Count; k++)
                    {
                        List<double> weightList = dbSelector.GetWeightsList(neuronsIDList[k]);

                        // Adding memory neuron data to text list:
                        string neuronDataText = $"layer_{i} neuron_{k} ";

                        for(int j = 0; j < weightList.Count; j++)
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

        #region Handling

        public double[] Handle(double[] data)
        {
            try
            {
                string handlingErrorText = "";
                double[] netResult =  _net.Handle(data, ref handlingErrorText);

                if (netResult == null)
                {
                    Logger.LogError(ErrorType.NonEqualsInputLengths, handlingErrorText);
                    return null;
                }
                else
                {
                    return netResult;
                }
            }
            catch
            {
                return null;
            }
        }

        #endregion
    }
}
