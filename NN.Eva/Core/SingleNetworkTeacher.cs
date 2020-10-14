using NN.Eva.Models;
using NN.Eva.Services;
using System;
using System.Collections.Generic;

namespace NN.Eva.Core
{
    public class SingleNetworkTeacher
    {
        public NeuralNetwork Network { get; set; }

        public NetworkStructure NetworkStructure { get; set; }

        public TrainingConfiguration TrainingConfiguration { get; set; }

        public List<double[]> InputDatasets { get; set; }

        public List<double[]> OutputDatasets { get; set; }

        public Logger Logger { get; set; }

        public bool LastTrainingSuccess { get; set; } = false;

        /// <summary>
        /// SafeMode. If true - additional checking for nullest results
        /// </summary>
        public bool SafeTrainingMode { get; set;} = true;

        private int Iteration = 0;

        /// <summary>
        /// Training
        /// </summary>
        public void Train()
        {
            if (Network == null) return;
            if (TrainingConfiguration == null) return;

            Iteration = TrainingConfiguration.EndIteration;
            if (Iteration == 0 || TrainingConfiguration.EndIteration - TrainingConfiguration.StartIteration <= 0) return;

            if (InputDatasets == null) return;
            if (OutputDatasets == null) return;

            if(SafeTrainingMode)
            {
                switch(TrainingConfiguration.TrainingAlgorithmType)
                {
                    case TrainingAlgorithmType.RProp:
                        SavedTrainingRProp();
                        break;
                    case TrainingAlgorithmType.BProp:
                    default:
                        SavedTrainingBProp();
                        break;
                }
            }
            else
            {
                switch (TrainingConfiguration.TrainingAlgorithmType)
                {
                    case TrainingAlgorithmType.RProp:
                        UnsafeTrainingRProp();
                        break;
                    case TrainingAlgorithmType.BProp:
                    default:
                        UnsafeTrainingBProp();
                        break;
                }
            }

            // Сохранение памяти сети:
            Network.SaveMemory(TrainingConfiguration.MemoryFolder + "//memory.txt", NetworkStructure);
        }

        #region Back propagation algorithm training

        private void SavedTrainingBProp()
        {
            for (int iteration = TrainingConfiguration.StartIteration; iteration < Iteration; iteration++)
            {
                // Calculating learn-speed rate:
                var learningSpeed = 0.01 * Math.Pow(0.1, (double)iteration / 150000);

                for (int k = 0; k < InputDatasets.Count; k++)
                {
                    string handlingErrorText = "";

                    // Handling:
                    double[] netResult = Network.Handle(InputDatasets[k], ref handlingErrorText);

                    if (netResult == null)
                    {
                        Logger.LogError(ErrorType.NonEqualsInputLengths, handlingErrorText);
                        return;
                    }

                    // Teaching:
                    try
                    {
                        Network.TeachBProp(InputDatasets[k], OutputDatasets[k], learningSpeed);
                    }
                    catch (Exception ex)
                    {
                        Logger.LogError(ErrorType.TrainError, ex);
                        return;
                    }
                }
            }

            // Запись события об успешном обучении:
            LastTrainingSuccess = true;
        }

        private void UnsafeTrainingBProp()
        {
            for (int iteration = TrainingConfiguration.StartIteration; iteration < Iteration; iteration++)
            {
                // Calculating learn-speed rate:
                var learningSpeed = 0.01 * Math.Pow(0.1, (double)iteration / 150000);

                for (int k = 0; k < InputDatasets.Count; k++)
                {
                    // Handling:
                    Network.HandleUnsafe(InputDatasets[k]);

                    // Teaching:
                    try
                    {
                        Network.TeachBProp(InputDatasets[k], OutputDatasets[k], learningSpeed);
                    }
                    catch (Exception ex)
                    {
                        Logger.LogError(ErrorType.TrainError, ex);
                        return;
                    }
                }
            }

            // Запись события об успешном обучении:
            LastTrainingSuccess = true;
        }

        #endregion

        #region Resilient propagation 

        private void SavedTrainingRProp()
        {
            // 1. Initialization training values:
            // 1.1. Initialize epoch error:
            double epochError = 1;

            // 1.2. Initialize gradients list (for weights updating)
            List<double[]> deltaList = InitializeNetworkRPropValues(0.1);

            List<double[]> gradientList = Network.GetGradients(epochError);

            List<double[]> updateValues = InitializeNetworkRPropValues(0);

            // 1.3. Initialize increasing & decreasing values:
            double increasingValue = 1.2;
            double decreasingValue = 0.5;

            // 2. Training:
            for (int iteration = TrainingConfiguration.StartIteration; iteration < Iteration; iteration++)
            {
                List<double[]> netLastEpochAnswers = new List<double[]>();

                // 2.1. Do one training epoch:
                for (int k = 0; k < InputDatasets.Count; k++)
                {
                    string handlingErrorText = "";

                    // Handling:
                    double[] netResult = Network.Handle(InputDatasets[k], ref handlingErrorText);

                    if (netResult == null)
                    {
                        Logger.LogError(ErrorType.NonEqualsInputLengths, handlingErrorText);
                        return;
                    }
                    
                    netLastEpochAnswers.Add(netResult);
                }

                // 2.2. Teaching:
                try
                {
                    epochError = RecalculateEpochError(netLastEpochAnswers);

                    // 2.2.1. Recalculating gradients:
                    List<double[]> newGradientList = new List<double[]>();

                    deltaList = RecalculateDeltaList(gradientList,
                                                     out newGradientList,
                                                     deltaList,
                                                     epochError,
                                                     increasingValue,
                                                     decreasingValue);

                    // Saving gradients of this epoch:
                    gradientList = newGradientList;

                    // 2.2.2. Teaching net:
                    // Calculating update-values:
                    for (int i = 0; i < deltaList.Count; i++)
                    {
                        for (int k= 0; k < deltaList[i].Length; k++)
                        {
                            if (gradientList[i][k] > 0)
                            {
                                updateValues[i][k] -= deltaList[i][k];
                            }

                            if (gradientList[i][k] < 0)
                            {
                                updateValues[i][k] += deltaList[i][k];
                            }
                            else
                            {
                                updateValues[i][k] = 0;
                            }
                        }
                    }

                    Network.TeachRProp(updateValues);
                }
                catch (Exception ex)
                {
                    Logger.LogError(ErrorType.TrainError, ex);
                    return;
                }
            }

            // Запись события об успешном обучении:
            LastTrainingSuccess = true;
        }

        private void UnsafeTrainingRProp()
        {
            // 1. Initialization training values:
            // 1.1. Initialize epoch error:
            double epochError = 1;

            // 1.2. Initialize gradients list (for weights updating)
            List<double[]> deltaList = InitializeNetworkRPropValues(0.1);

            List<double[]> gradientList = Network.GetGradients(epochError);

            List<double[]> updateValues = InitializeNetworkRPropValues(0);

            // 1.3. Initialize increasing & decreasing values:
            double increasingValue = 1.2;
            double decreasingValue = 0.5;

            // 2. Training:
            for (int iteration = TrainingConfiguration.StartIteration; iteration < Iteration; iteration++)
            {
                List<double[]> netLastEpochAnswers = new List<double[]>();

                // 2.1. Do one training epoch:
                for (int k = 0; k < InputDatasets.Count; k++)
                {
                    // Handling & saving results:
                    netLastEpochAnswers.Add(Network.HandleUnsafe(InputDatasets[k]));
                }

                // 2.2. Teaching:
                try
                {
                    epochError = RecalculateEpochError(netLastEpochAnswers);

                    // 2.2.1. Recalculating gradients:
                    List<double[]> newGradientList = new List<double[]>();

                    deltaList = RecalculateDeltaList(gradientList,
                                                     out newGradientList,
                                                     deltaList,
                                                     epochError,
                                                     increasingValue,
                                                     decreasingValue);

                    // Saving gradients of this epoch:
                    gradientList = newGradientList;

                    // 2.2.2. Teaching net:
                    // Calculating update-values:
                    for (int i = 0; i < deltaList.Count; i++)
                    {
                        for (int k = 0; k < deltaList[i].Length; k++)
                        {
                            if (gradientList[i][k] > 0)
                            {
                                updateValues[i][k] -= deltaList[i][k];
                            }

                            if (gradientList[i][k] < 0)
                            {
                                updateValues[i][k] += deltaList[i][k];
                            }
                            else
                            {
                                updateValues[i][k] = 0;
                            }
                        }
                    }

                    Network.TeachRProp(updateValues);
                }
                catch (Exception ex)
                {
                    Logger.LogError(ErrorType.TrainError, ex);
                    return;
                }
            }

            // Запись события об успешном обучении:
            LastTrainingSuccess = true;
        }

        private List<double[]> InitializeNetworkRPropValues(double initializeValues)
        {
            List<double[]> deltaList = new List<double[]>();

            // Initialize first layers gradient values:
            double[] deltaFirstRow = new double[NetworkStructure.InputVectorLength];

            for (int i = 0; i < NetworkStructure.InputVectorLength; i++)
            {
                deltaFirstRow[i] = initializeValues;
            }

            deltaList.Add(deltaFirstRow);

            // Initialize first layers gradient values:
            for (int i = 0; i < NetworkStructure.NeuronsByLayers.Length; i++)
            {
                double[] deltaRowTemp = new double[NetworkStructure.NeuronsByLayers[i]];

                for (int k = 0; k < NetworkStructure.NeuronsByLayers[i]; k++)
                {
                    deltaRowTemp[k] = initializeValues;
                }

                deltaList.Add(deltaRowTemp);
            }

            return deltaList;
        }

        private double RecalculateEpochError(List<double[]> netLastAnswers)
        {
            double sum = 0;

            for (int i = 0; i < netLastAnswers.Count; i++)
            {
                for (int k = 0; k < netLastAnswers[i].Length; k++)
                {
                    sum += Math.Pow(netLastAnswers[i][k] - OutputDatasets[i][k], 2);
                }
            }

            return 0.5 * sum;
        }

        private List<double[]> RecalculateDeltaList(List<double[]> lastGradientList,
                                                    out List<double[]> newGradientList,
                                                    List<double[]> deltaList,
                                                    double epochError,
                                                    double increasingValue,
                                                    double decreasingValue)
        {
            newGradientList = Network.GetGradients(epochError);

            // Recalculating gradients:
            for (int i = 0; i < newGradientList.Count; i++)
            {
                for (int k = 0; k < newGradientList[i].Length; k++)
                {
                    double gradientChanging = lastGradientList[i][k] * newGradientList[i][k];

                    if (gradientChanging > 0)
                    {
                        deltaList[i][k] = increasingValue * deltaList[i][k];
                    }

                    if (gradientChanging < 0)
                    {
                        deltaList[i][k] = decreasingValue * deltaList[i][k];
                    }
                    else
                    {
                        deltaList[i][k] = 0;
                    }
                }
            }

            return deltaList;
        }

        #endregion
    }
}
