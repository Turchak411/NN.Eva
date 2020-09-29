using MySql.Data.MySqlClient;
using NN.Eva.Models;
using NN.Eva.Services;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using NN.Eva.Core.Database;
using NN.Eva.Models.Database;

namespace NN.Eva.Core
{
    public class NetworksTeacher
    {
        private List<NeuralNetwork> _netsList;

        private NetworkStructure _netStructure;

        /// <summary>
        /// Services
        /// </summary>
        private FileManager _fileManager;
        private MemoryChecker _memoryChecker;
        private Logger _logger;

        /// <summary>
        /// Global current iterations
        /// </summary>
        public int Iteration { get; set; } = 0;

        /// <summary>
        /// Testing objects
        /// </summary>
        public List<TrainObject> TestVectors { get; set; }

        public NetworksTeacher(NetworkStructure netStructure, int netsCount, FileManager fileManager)
        {
            _netStructure = netStructure;

            _netsList = new List<NeuralNetwork>();

            _fileManager = fileManager;
            _logger = new Logger();

            try
            {
                // Ицициализация сети по одинаковому шаблону:
                for (int i = 0; i < netsCount; i++)
                {
                    _netsList.Add(new NeuralNetwork(netStructure.InputVectorLength,
                        netStructure.NeuronsByLayers,
                        fileManager, "memory_" + i + ".txt"));
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ErrorType.MemoryInitializeError, ex);
            }
        }

        #region Trained testing

        public void CommonTest()
        {
            if (TestVectors == null) return;

            var result = new StringBuilder();
            TestVectors.ForEach(vector => result.Append($"   {vector._content}     "));
            result.Append('\n');

            for (int i = 0; i < _netsList.Count; i++)
            {
                for (int k = 0; k < TestVectors.Count; k++)
                {
                    // Получение ответа:
                    var outputVector = _netsList[i].Handle(TestVectors[k]._vectorValues);

                    result.Append($"{outputVector[0]:f5}\t");
                }
                result.Append('\n');
            }

            Console.WriteLine(result);
        }

        public void CommonTestColorized()
        {
            if (TestVectors == null) return;

            var result = new StringBuilder();
            TestVectors.ForEach(vector => result.Append($"   {vector._content}     "));
            result.Append('\n');

            for (int i = 0; i < _netsList.Count; i++)
            {
                for (int k = 0; k < TestVectors.Count; k++)
                {
                    // Получение ответа:
                    var outputVector = _netsList[i].Handle(TestVectors[k]._vectorValues);

                    try
                    {
                        // Костыль: для корректного теста сетям нужна по крайней мере одна итерация обучения:
                        _netsList[i].Teach(TestVectors[k]._vectorValues, new double[1] { 1 }, 0.01); //0.000000000000001);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ErrorType.TrainError, ex);
                    }

                    Console.ForegroundColor = GetColorByActivation(outputVector[0]);
                    Console.Write($"{outputVector[0]:f5}\t");
                }

                Console.ForegroundColor = ConsoleColor.Gray;
                Console.Write('\n');
            }

            Console.ForegroundColor = ConsoleColor.Gray;
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

        public void PrintLearnStatistic(TrainingConfiguration trainingConfig, bool withLogging = false)
        {
            Console.WriteLine("Start calculating statistic...");

            int testPassed = 0;
            int testFailed = 0;
            int testFailed_lowActivationCause = 0;

            #region Load data from file

            List<double[]> inputDataSets;
            List<double[]> outputDataSets;

            try
            {
                inputDataSets = _fileManager.LoadTrainingDataset(trainingConfig.InputDatasetFilename);
                outputDataSets = _fileManager.LoadTrainingDataset(trainingConfig.OutputDatasetFilename);
            }
            catch (Exception ex)
            {
                _logger.LogError(ErrorType.SetMissing, ex);
                return;
            }

            #endregion

            for (int i = 0; i < inputDataSets.Count; i++)
            {
                List<double> netResults = new List<double>();

                for (int k = 0; k < _netsList.Count; k++)
                {
                    // Получение ответа:
                    netResults.Add(_netsList[k].Handle(inputDataSets[i])[0]);
                }

                // Поиск максимально активирующейся сети (класса) с заданным порогом активации:
                int maxIndex = FindMaxIndex(netResults, 0.8);

                if (maxIndex == -1)
                {
                    testFailed++;
                    testFailed_lowActivationCause++;
                }
                else
                {
                    if (outputDataSets[i][maxIndex] != 1)
                    {
                        testFailed++;
                    }
                    else
                    {
                        testPassed++;
                    }
                }
            }

            // Logging (optional):
            if (withLogging)
            {
                _logger.LogTrainResults(testPassed, testFailed, testFailed_lowActivationCause, Iteration);
            }

            Console.WriteLine("Test passed: {0}\nTest failed: {1}\n     - Low activation causes: {2}\nPercent learned: {3:f2}", testPassed,
                                                                                                                           testFailed,
                                                                                                                           testFailed_lowActivationCause,
                                                                                                                           (double)testPassed * 100 / (testPassed + testFailed));
        }

        private int FindMaxIndex(List<double> netResults, double threshold = 0.8)
        {
            int maxIndex = -1;
            double maxValue = -1;

            for (int i = 0; i < netResults.Count; i++)
            {
                if (maxValue < netResults[i] && netResults[i] >= threshold)
                {
                    maxIndex = i;
                    maxValue = netResults[i];
                }
            }

            return maxIndex;
        }

        #endregion

        public bool CheckMemory(string memoryFolder = "Memory")
        {
            bool isValid = true;

            Console.WriteLine("Start memory cheking...");

            _memoryChecker = new MemoryChecker();

            for (int i = 0; i < _netsList.Count; i++)
            {
                bool isCurrentNetMemoryValid = _netStructure == null
                    ? _memoryChecker.IsValidQuickCheck(memoryFolder + "//memory_" + i + ".txt")
                    : _memoryChecker.IsValid(memoryFolder + "//memory_" + i + ".txt", _netStructure);

                if (isCurrentNetMemoryValid)
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine("memory_" + i + " - is valid.");
                }
                else
                {
                    isValid = false;
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("memory_" + i + " - is invalid!");
                }
            }

            Console.ForegroundColor = ConsoleColor.Gray;

            return isValid;
        }

        public void BackupMemory(string memoryFolder = "Memory", string backupsDirectoryName = ".memory_backups",
                                 DatabaseConfig dbConfig = null, string networkStructureInfo = "no information")
        {
            // Check for existing main backups-directory:
            if (!Directory.Exists(memoryFolder + "//" + backupsDirectoryName))
            {
                Directory.CreateDirectory(memoryFolder + "//" + backupsDirectoryName);
            }

            // Check for already-existing sub-directory (trainCount-named):
            if (!Directory.Exists(memoryFolder + "//" + backupsDirectoryName + "//" + Iteration))
            {
                Directory.CreateDirectory(memoryFolder + "//" + backupsDirectoryName + "//" + Iteration);
            }

            // Saving memory:
            for (int i = 0; i < _netsList.Count; i++)
            {
                _netsList[i].SaveMemory(memoryFolder + "//" + backupsDirectoryName + "//" + Iteration + "//memory_" + i + ".txt", _netStructure);
            }

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

        #region Training

        /// <summary>
        /// Обучение сети
        /// </summary>
        /// <param name="startIteration"></param>
        /// <param name="withSort"></param>
        public void TrainNets(TrainingConfiguration trainingConfig, int iterationsToPause)
        {
            Iteration = trainingConfig.EndIteration;

            #region Load data from file

            List<double[]> inputDataSets;
            List<double[]> outputDataSets;

            try
            {
                inputDataSets = _fileManager.LoadTrainingDataset(trainingConfig.InputDatasetFilename);
                outputDataSets = _fileManager.LoadTrainingDataset(trainingConfig.OutputDatasetFilename);
            }
            catch (Exception ex)
            {
                _logger.LogError(ErrorType.SetMissing, ex);
                return;
            }

            #endregion

            Console.WriteLine("Training start...");
            try
            {
                SingleNetworkTeacher[] netTeachers = new SingleNetworkTeacher[_netsList.Count];

                List<TrainingConfiguration> trainingConfigs = InitializeTrainingSubConfigs(trainingConfig, iterationsToPause);

                // Initialize teachers:
                for (int i = 0; i < netTeachers.Length; i++)
                {
                    netTeachers[i] = new SingleNetworkTeacher
                    {
                        Id = i,
                        Network = _netsList[i],
                        NetworkStructure = _netStructure,
                        TrainingConfiguration = trainingConfig,
                        InputDatasets = inputDataSets,
                        OutputDatasets = outputDataSets,
                        Logger = _logger
                    };
                }

                List<Thread> threadList;

                // Iteration multithreading train:
                for (int j = 0; j < trainingConfigs.Count; j++)
                {
                    threadList = new List<Thread>();

                    for (int i = 0; i < netTeachers.Length; i++)
                    {
                        threadList.Add(new Thread(netTeachers[i].Train));
                        threadList[i].Start();
                    }

                    Wait(threadList);

                    if (j != trainingConfigs.Count - 1)
                    {
                        Console.WriteLine("Iterations already finished: " + iterationsToPause * (j + 1));
                    }
                    else
                    {
                        Console.WriteLine("Iterations already finished: " + trainingConfig.EndIteration);
                    }

                    CommonTestColorized();
                }

                Console.WriteLine("Training success!");
            }
            catch (Exception ex)
            {
                _logger.LogError(ErrorType.TrainError, ex);
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
                        StartIteration = currentIterPosition,
                        EndIteration = currentIterPosition + iterationsToPause,
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
                        StartIteration = currentIterPosition,
                        EndIteration = trainingConfig.EndIteration,
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

        private void Wait(List<Thread> threadList)
        {
            while (true)
            {
                int WorkCount = 0;

                for (int i = 0; i < threadList.Count; i++)
                {
                    WorkCount += (threadList[i].IsAlive) ? 0 : 1;
                }

                if (WorkCount == threadList.Count) break;
            }
        }

        #endregion

        #region Database

        private void SavingMemoryToDB(DatabaseConfig dbConfig, string networkStructure, Guid userId)
        {
            Logger logger = new Logger();

            try
            {
                DBInserter dbInserter = new DBInserter(logger, dbConfig);

                // Saving networks info:
                for (int i = 0; i < _netsList.Count; i++)
                {
                    _netsList[i].SaveMemoryToDB(Iteration, networkStructure, userId, dbInserter);
                    Console.WriteLine("Network #{0} backuped successfully!", i);
                }
            }
            catch (Exception ex)
            {
                logger.LogError(ErrorType.DBInsertError, "Save memory to database error!\n" + ex);
                Console.WriteLine($" {DateTime.Now} Save memory to database error!\n {ex}");
            }
        }

        public void DBMemoryAbort(DatabaseConfig dbConfig)
        {
            Logger logger = new Logger();

            try
            {
                DBDeleter dbDeleter = new DBDeleter(logger, dbConfig);

                // Aborting saving network's info:
                for (int i = 0; i < _netsList.Count; i++)
                {
                    _netsList[i].DBMemoryAbort(dbDeleter);
                    Console.WriteLine("Another network's memory backup aborted successfully!");
                }
            }
            catch (Exception ex)
            {
                logger.LogError(ErrorType.DBDeleteError, "DB Memory backup aborting error!\n" + ex);
                Console.WriteLine($" {DateTime.Now } DB Memory backup aborting error!\n {ex}");
            }
        }

        public void DBMemoryLoad(DatabaseConfig dbConfig, Guid networkID, string destinationMemoryFilePath)
        {
            Logger logger = new Logger();

            try
            {
                DBSelector dbSelector = new DBSelector(logger, dbConfig);

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
                _fileManager.SaveMemoryFromModel(memoryInTextList, destinationMemoryFilePath);
            }
            catch (Exception ex)
            {
                logger.LogError(ErrorType.DBMemoryLoadError, "DB Memory loading error!\n" + ex);
                Console.WriteLine($" {DateTime.Now } DB Memory loading error!\n {ex}");
            }
        }

        #endregion

        #region Handling

        public double[] HandleAsAssembly(double[] data)
        {
            try
            {
                double[] resultVector = new double[_netsList.Count];

                for (int i = 0; i < _netsList.Count; i++)
                {
                    resultVector[i] = _netsList[i].Handle(data)[0];
                }

                return resultVector;
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
                return _netsList[0].Handle(data);
            }
            catch
            {
                return null;
            }
        }

        #endregion
    }
}
