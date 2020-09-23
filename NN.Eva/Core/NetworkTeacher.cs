using MySql.Data.MySqlClient;
using NN.Eva.Models;
using NN.Eva.Services;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using NN.Eva.Core.Database;

namespace NN.Eva.Core
{
    public class NetworksTeacher
    {
        private List<NeuralNetwork> _netsList;

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

        public NetworksTeacher(NetworkStructure netStructure, int netsCount, FileManager fileManager, string memoryFolderName = "Memory")
        {
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
                        fileManager, memoryFolderName + "//memory_" + i + ".txt"));
                    // TODO: Сделать загрузку готовой памяти из базы данных (реализация DBSelector'а)
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ErrorType.MemoryInitializeError, ex);
            }
        }

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

        public void PrintLearnStatistic(TrainConfiguration trainConfig, bool withLogging = false)
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
                inputDataSets = _fileManager.LoadSingleDataset(trainConfig.InputDatasetFilename);
                outputDataSets = _fileManager.LoadSingleDataset(trainConfig.OutputDatasetFilename);
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

        public bool CheckMemory(string memoryFolder = "")
        {
            bool isValid = true;

            Console.WriteLine("Start memory cheking...");

            _memoryChecker = new MemoryChecker();

            for (int i = 0; i < _netsList.Count; i++)
            {
                if (_memoryChecker.IsValid(memoryFolder + "//" + "memory_" + i + ".txt"))
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

        public void BackupMemory(string memoryFolder = "", string backupsDirectoryName = ".memory_backups",
                                 MySqlConnection dbConnection = null, string networkStructure = "no information")
        {
            // Check for existing main backups-directory:
            if (!Directory.Exists(memoryFolder + "//" + backupsDirectoryName))
            {
                Directory.CreateDirectory(memoryFolder + "//" + backupsDirectoryName);
            }

            // Check for already-existing sub-directory (trainCount-named):
            if (!Directory.Exists(memoryFolder + "//" + backupsDirectoryName + "/ " + Iteration))
            {
                Directory.CreateDirectory(memoryFolder + "//" + backupsDirectoryName + "/" + Iteration);
            }

            // Saving memory:
            for (int i = 0; i < _netsList.Count; i++)
            {
                _netsList[i].SaveMemory(memoryFolder + "//" + backupsDirectoryName + "/" + Iteration + "/memory_" + i + ".txt");
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
            if (dbConnection != null)
            {
                Console.WriteLine("Backuping memory to database...");

                SavingMemoryToDB(dbConnection, networkStructure, userId);
            }

            Console.WriteLine("Memory backuped!");
        }

        /// <summary>
        /// Обучение сети
        /// </summary>
        /// <param name="startIteration"></param>
        /// <param name="withSort"></param>
        public void TrainNets(TrainConfiguration trainConfig, int iterationsToPause)
        {
            Iteration = trainConfig.EndIteration;

            #region Load data from file

            List<double[]> inputDataSets;
            List<double[]> outputDataSets;

            try
            {
                inputDataSets = _fileManager.LoadSingleDataset(trainConfig.InputDatasetFilename);
                outputDataSets = _fileManager.LoadSingleDataset(trainConfig.OutputDatasetFilename);
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

                List<TrainConfiguration> trainConfigs = InitializeTrainConfigs(trainConfig, iterationsToPause);

                // Initialize teachers:
                for (int i = 0; i < netTeachers.Length; i++)
                {
                    netTeachers[i] = new SingleNetworkTeacher
                    {
                        Id = i,
                        Network = _netsList[i],
                        TrainConfiguration = trainConfig,
                        InputDatasets = inputDataSets,
                        OutputDatasets = outputDataSets,
                        Logger = _logger
                    };
                }

                List<Thread> threadList;

                // Iteration multithreading train:
                for (int j = 0; j < trainConfigs.Count; j++)
                {
                    threadList = new List<Thread>();

                    for (int i = 0; i < netTeachers.Length; i++)
                    {
                        threadList.Add(new Thread(netTeachers[i].Train));
                        threadList[i].Start();
                    }

                    Wait(threadList);

                    if (j != trainConfigs.Count - 1)
                    {
                        Console.WriteLine("Iterations already finished: " + iterationsToPause * (j + 1));
                    }
                    else
                    {
                        Console.WriteLine("Iterations already finished: " + trainConfig.EndIteration);
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

        private List<TrainConfiguration> InitializeTrainConfigs(TrainConfiguration trainConfig, int iterationsToPause)
        {
            List<TrainConfiguration> trainConfigs = new List<TrainConfiguration>();

            int currentIterPosition = trainConfig.StartIteration;
            while (true)
            {
                if (trainConfig.EndIteration - currentIterPosition - 1 >= iterationsToPause)
                {
                    var trainConfigItem = new TrainConfiguration()
                    {
                        StartIteration = currentIterPosition,
                        EndIteration = currentIterPosition + iterationsToPause,
                        InputDatasetFilename = trainConfig.InputDatasetFilename,
                        OutputDatasetFilename = trainConfig.OutputDatasetFilename
                    };

                    trainConfigs.Add(trainConfigItem);

                    currentIterPosition += iterationsToPause;
                }
                else
                {
                    var trainConfigItem = new TrainConfiguration()
                    {
                        StartIteration = currentIterPosition,
                        EndIteration = trainConfig.EndIteration,
                        InputDatasetFilename = trainConfig.InputDatasetFilename,
                        OutputDatasetFilename = trainConfig.OutputDatasetFilename
                    };

                    trainConfigs.Add(trainConfigItem);

                    break;
                }
            }

            Console.WriteLine("Train configuration object created!");

            return trainConfigs;
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

        private void SavingMemoryToDB(MySqlConnection dbConnection, string networkStructure, Guid userId)
        {
            Logger logger = new Logger();

            try
            {
                using (dbConnection)
                {
                    dbConnection.Open();

                    DBInserter dbInserter = new DBInserter(dbConnection, logger);

                    // Saving networks info:
                    for (int i = 0; i < _netsList.Count; i++)
                    {
                        _netsList[i].SaveMemoryToDB(Iteration, networkStructure, userId, dbInserter);
                        Console.WriteLine("Network #{0} backuped successfully!", i);
                    }

                    dbConnection.Close();
                }
            }
            catch (Exception ex)
            {
                logger.LogError(ErrorType.DBInsertError, "Save memory to database error!\n" + ex);
                Console.WriteLine($" {DateTime.Now } Save memory to database error!\n {ex}");
            }
        }

        public void DBMemoryAbort(MySqlConnection dbConnection)
        {
            Logger logger = new Logger();

            try
            {
                using (dbConnection)
                {
                    dbConnection.Open();

                    DBDeleter dbDeleter = new DBDeleter(dbConnection, logger);

                    // Aborting saving network's info:
                    for (int i = 0; i < _netsList.Count; i++)
                    {
                        _netsList[i].DBMemoryAbort(dbDeleter);
                        Console.WriteLine("Another network's memory backup aborted successfully!");
                    }

                    dbConnection.Close();
                }
            }
            catch (Exception ex)
            {
                logger.LogError(ErrorType.DBInsertError, "DB Memory backup aborting error!\n" + ex);
                Console.WriteLine($" {DateTime.Now } DB Memory backup aborting error!\n {ex}");
            }
        }

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
    }
}
