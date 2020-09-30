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
        private NeuralNetwork _net;

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

        public NetworksTeacher(NetworkStructure netStructure, FileManager fileManager)
        {
            _netStructure = netStructure;

            _fileManager = fileManager;
            _logger = new Logger();

            try
            {
                // Ицициализация сети по одинаковому шаблону:
                _net = new NeuralNetwork(netStructure.InputVectorLength,
                        netStructure.NeuronsByLayers,
                        fileManager, "memory.txt");
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

            for (int k = 0; k < TestVectors.Count; k++)
            {
                // Получение ответа:
                var outputVector = _net.Handle(TestVectors[k]._vectorValues);

                result.Append($"{outputVector[0]:f5}\t");
            }
            result.Append('\n');

            Console.WriteLine(result);
        }

        public void CommonTestColorized()
        {
            if (TestVectors == null) return;

            var result = new StringBuilder();
            TestVectors.ForEach(vector => result.Append($"   {vector._content}     "));
            result.Append('\n');

            for (int k = 0; k < TestVectors.Count; k++)
            {
                // Получение ответа:
                var outputVector = _net.Handle(TestVectors[k]._vectorValues);

                try
                {
                    // Костыль: для корректного теста сетям нужна по крайней мере одна итерация обучения:
                    _net.Teach(TestVectors[k]._vectorValues, new double[1] { 1 }, 0.01); //0.000000000000001);
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
                // Получение ответа:
                double[] netResult = _net.Handle(inputDataSets[i]);

                if (IsVectorsRoughlyEquals(outputDataSets[i], netResult, 0.3))
                {
                    testPassed++;
                }
                else
                {
                    testFailed++;
                }
            }

            // Logging (optional):
            if (withLogging)
            {
                _logger.LogTrainResults(testPassed, testFailed, Iteration);
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
                // TODO: Сделать универсальную формулу подсчета:
                if (controlVector1[i] < sourceVector0[i] - equalsPercent || controlVector1[i] > sourceVector0[i] + equalsPercent)
                {
                    return false;
                }
            }

            return true;
        }

        #endregion

        public bool CheckMemory(string memoryFolder = "Memory")
        {
            bool isValid = true;

            Console.WriteLine("Start memory cheking...");

            _memoryChecker = new MemoryChecker();

            bool isCurrentNetMemoryValid = _netStructure == null
                ? _memoryChecker.IsValidQuickCheck(memoryFolder + "//memory.txt")
                : _memoryChecker.IsValid(memoryFolder + "//memory.txt", _netStructure) &&
                  _fileManager.IsMemoryEqualsDefault("memory.txt");

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
            _net.SaveMemory(memoryFolder + "//" + backupsDirectoryName + "//" + Iteration + "//memory.txt", _netStructure);

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
        public void TrainNet(TrainingConfiguration trainingConfig, int iterationsToPause)
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
                List<TrainingConfiguration> trainingConfigs = InitializeTrainingSubConfigs(trainingConfig, iterationsToPause);

                // Initialize teachers:
                SingleNetworkTeacher netTeacher = new SingleNetworkTeacher
                {
                    Network = _net,
                    NetworkStructure = _netStructure,
                    TrainingConfiguration = trainingConfig,
                    InputDatasets = inputDataSets,
                    OutputDatasets = outputDataSets,
                    Logger = _logger
                };

                // Iteration multithreading train:
                for (int j = 0; j < trainingConfigs.Count; j++)
                {
                    netTeacher.TrainingConfiguration = trainingConfigs[j];

                    Thread thread = new Thread(netTeacher.Train);
                    thread.Start();
                    Wait(thread);

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

                // TODO:
                // ПОлучение обученной сети:
                _net = netTeacher.Network;

                Console.WriteLine("Training success!");
            }
            catch (Exception ex)
            {
                _logger.LogError(ErrorType.TrainError, ex);
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

        #region Database

        private void SavingMemoryToDB(DatabaseConfig dbConfig, string networkStructure, Guid userId)
        {
            Logger logger = new Logger();

            try
            {
                DBInserter dbInserter = new DBInserter(logger, dbConfig);

                // Saving networks info:
                _net.SaveMemoryToDB(Iteration, networkStructure, userId, dbInserter);
                    Console.WriteLine("Networks memory backuped successfully!");
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
                _net.DBMemoryAbort(dbDeleter);
                    Console.WriteLine("Another network's memory backup aborted successfully!");
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

        public double[] Handle(double[] data)
        {
            try
            {
                return _net.Handle(data);
            }
            catch
            {
                return null;
            }
        }

        #endregion
    }
}
