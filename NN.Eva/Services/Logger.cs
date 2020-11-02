using NN.Eva.Models;
using System;
using System.IO;
using System.Management;

namespace NN.Eva.Services
{
    public static class Logger
    {
        private static string TrainLogsDirectoryName { get; set; } = ".logs\\trainingLogs";

        private static string ErrorLogsDirectoryName { get; set; } = ".logs\\errorLogs";

        public static void LogTrainingResults(int testPassed, int testFailed, TrainingConfiguration trainConfig, string elapsedTime)
        {
            // Check for existing this logs-directory:
            if (!Directory.Exists(TrainLogsDirectoryName))
            {
                Directory.CreateDirectory(TrainLogsDirectoryName);
            }

            // Save log:
            using (StreamWriter fileWriter = new StreamWriter(TrainLogsDirectoryName + "/" + trainConfig.EndIteration + ".txt"))
            {
                fileWriter.WriteLine("Test passed: " + testPassed);
                fileWriter.WriteLine("Test failed: " + testFailed);
                fileWriter.WriteLine("Percent learned: {0:f2}", (double)testPassed * 100 / (testPassed + testFailed));
                
                // Writing additional training data:
                fileWriter.WriteLine("\n============================\n");

                // Writing elapsed time: 
                if(elapsedTime != "")
                {
                    fileWriter.WriteLine("Time spend: " + elapsedTime);
                    fileWriter.WriteLine("Start iteration: " + trainConfig.StartIteration);
                    fileWriter.WriteLine("End iteration: " + trainConfig.EndIteration);
                    fileWriter.WriteLine("============================\n");
                }

                // Writing system characteristics:
                fileWriter.WriteLine("System characteristics:");
                fileWriter.WriteLine("OS Version: {0} {1}", Environment.OSVersion, Environment.Is64BitOperatingSystem ? "64 bit" : "32 bit");
                fileWriter.WriteLine("Processors count: " + Environment.ProcessorCount);

                ManagementClass myManagementClass = new ManagementClass("Win32_Processor");
                PropertyDataCollection myProperties = myManagementClass.Properties;

                foreach (var myProperty in myProperties)
                {
                    switch (myProperty.Name)
                    {
                        case "CurrentClockSpeed":
                        case "Name":
                            fileWriter.WriteLine($"{myProperty.Name}: { myProperty.Value ?? "-"}");
                            break;
                    }
                }
            }

            Console.WriteLine("Learn statistic logs saved in {0}!", TrainLogsDirectoryName);
        }

        public static void LogWarning(WarningType warningType)
        {
            WriteLog(GetWarningTextByType(warningType));
        }

        public static void LogError(ErrorType errorType, Exception ex)
        {
            WriteLog(GetErrorTextByType(errorType), ex.Message);
        }

        public static void LogError(ErrorType errorType, string errorText = "")
        {
            WriteLog(GetErrorTextByType(errorType), errorText);
        }

        private static string GetWarningTextByType(WarningType warningType)
        {
            string warningText;

            switch (warningType)
            {
                case WarningType.UsingUnsafeTrainingMode:
                    warningText = "You are using unsafe training mode! Make sure that your input-vectors lengths are equals with Network's ones!";
                    break;
                default:
                    warningText = "";
                    break;
            }

            Console.ForegroundColor = ConsoleColor.DarkYellow;
            Console.WriteLine("[WARNING] " + warningText);
            Console.ForegroundColor = ConsoleColor.Gray;

            return warningText + "\n";
        }

        private static string GetErrorTextByType(ErrorType errorType)
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
                case ErrorType.MemoryGenerateError:
                    errorText = "Memory generate error!";
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
                case ErrorType.DBMemoryLoadError:
                    errorText = "Database loading error!";
                    break;
                case ErrorType.NonEqualsInputLengths:
                    errorText = "Length of network's input vector and length of training dataset's vector is not equals!";
                    break;
                case ErrorType.OperationWithNonexistentNetwork:
                    errorText = "Error in operation with nonexistent network! Please, create the Network first!";
                    break;
                case ErrorType.UnknownError:
                default:
                    errorText = "Unknown error!";
                    break;
            }

            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("[ERROR] " + errorText);
            Console.ForegroundColor = ConsoleColor.Gray;

            return errorText + "\n";
        }

        private static void WriteLog(string systemErrorText, string additionalErrorText = "")
        {
            // Check for existing this logs - directory:
            if (!Directory.Exists(ErrorLogsDirectoryName))
            {
                Directory.CreateDirectory(ErrorLogsDirectoryName);
            }

            // Save log:
            try
            {
                using (StreamWriter fileWriter = new StreamWriter(ErrorLogsDirectoryName + "/"
                                                                                          + DateTime.Now.Day + "_"
                                                                                          + DateTime.Now.Month + "_"
                                                                                          + DateTime.Now.Year + "_"
                                                                                          + DateTime.Now.Ticks +
                                                                                          ".txt"))
                {
                    fileWriter.WriteLine("\nTime: " + DateTime.Now);
                    fileWriter.WriteLine("System error text: " + systemErrorText);

                    if (additionalErrorText != "")
                    {
                        fileWriter.Write("Additional error text: " + additionalErrorText);
                    }
                }

                Console.WriteLine("Error log saved in {0}!", ErrorLogsDirectoryName);
            }
            catch { }
        }
    }
}
