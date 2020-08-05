using NN.Eva.Models;
using System;
using System.IO;

namespace NN.Eva.Services
{
    public class Logger
    {
        private string _trainLogsDirectoryName;
        private string _errorLogsDirectoryName;

        public Logger()
        {
            _trainLogsDirectoryName = ".logs/trainLogs";
            _errorLogsDirectoryName = ".logs/errorLogs";
        }

        public Logger(string trainLogsDirectoryName, string errorLogsDirectoryName)
        {
            _trainLogsDirectoryName = trainLogsDirectoryName;
            _errorLogsDirectoryName = errorLogsDirectoryName;
        }

        public void LogTrainResults(int testPassed, int testFailed, int testFailedLowActivationCause, int iteration)
        {
            // Check for existing this logs-directory:
            if (!Directory.Exists(_trainLogsDirectoryName))
            {
                Directory.CreateDirectory(_trainLogsDirectoryName);
            }

            // Save log:
            using (StreamWriter fileWriter = new StreamWriter(_trainLogsDirectoryName + "/" + iteration + ".txt"))
            {
                fileWriter.WriteLine("Test passed: " + testPassed);
                fileWriter.WriteLine("Test failed: " + testFailed);
                fileWriter.WriteLine("     - Low activation causes: " + testFailedLowActivationCause);
                fileWriter.WriteLine("Percent learned: {0:f2}", (double)testPassed * 100 / (testPassed + testFailed));
            }

            Console.WriteLine("Learn statistic logs saved in {0}!", _trainLogsDirectoryName);
        }

        public void LogError(ErrorType errorType, Exception ex)
        {
            WriteError(GetErrorTextByType(errorType) + ex);
        }

        public void LogError(ErrorType errorType, string errorText = "")
        {
            WriteError(GetErrorTextByType(errorType) + errorText);
        }

        private string GetErrorTextByType(ErrorType errorType)
        {
            string errorText;

            switch (errorType)
            {
                case ErrorType.MemoryMissing:
                    errorText = "Memory is missing!";
                    break;
                case ErrorType.MemoryInitializeError:
                    errorText = "Memory initialize error! Check compliance the network structure and the memory-file!";
                    break;
                case ErrorType.SetMissing:
                    errorText = "One or more train set is missing!";
                    break;
                case ErrorType.TrainError:
                    errorText = "Network training error!";
                    break;
                case ErrorType.DBConnectionError:
                    errorText = "Database connection error!";
                    break;
                case ErrorType.DBInsertError:
                    errorText = "Database inserting error!";
                    break;
                case ErrorType.DBDeleteError:
                    errorText = "Database deleting error!";
                    break;
                default:
                    errorText = "";
                    break;
            }

            Console.WriteLine(errorText + "\n");
            return errorText + "\n";
        }

        private void WriteError(string error)
        {
            // Check for existing this logs - directory:
            if (!Directory.Exists(_errorLogsDirectoryName))
            {
                Directory.CreateDirectory(_errorLogsDirectoryName);
            }

            // Save log:
            try
            {
                using (StreamWriter fileWriter = new StreamWriter(_errorLogsDirectoryName + "/"
                                                                                          + DateTime.Now.Day + "_"
                                                                                          + DateTime.Now.Month + "_"
                                                                                          + DateTime.Now.Year + "_"
                                                                                          + DateTime.Now.Ticks +
                                                                                          ".txt"))
                {
                    fileWriter.WriteLine("\nTime: " + DateTime.Now);
                    fileWriter.WriteLine(error);
                }

                Console.WriteLine("Error log saved in {0}!", _errorLogsDirectoryName);
            }
            catch { }
        }
    }
}
