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
                StartSavedTraining(TrainingConfiguration.TrainingAlgorithmType);
            }
            else
            {
                StartUnsafeTraining(TrainingConfiguration.TrainingAlgorithmType);
            }

            // Сохранение памяти сети:
            Network.SaveMemory(TrainingConfiguration.MemoryFolder + "//memory.txt", NetworkStructure);
        }

        private void StartSavedTraining(TrainingAlgorithmType trainingAlgorithm)
        {
            switch (trainingAlgorithm)
            {
                //case TrainingAlgorithmType.RProp:
                //    SavedTrainingRProp();
                //    break;
                case TrainingAlgorithmType.BProp:
                default:
                    SavedTrainingBProp();
                    break;
            }
        }

        private void StartUnsafeTraining(TrainingAlgorithmType trainingAlgorithm)
        {
            switch (trainingAlgorithm)
            {
                //case TrainingAlgorithmType.RProp:
                //    UnsafeTrainingRProp();
                //    break;
                case TrainingAlgorithmType.BProp:
                default:
                    UnsafeTrainingBProp();
                    break;
            }
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

            // 1.2. Initialize gradients list:
            List<double[]> lastGradientList = InitializeNetworkRPropValues(0);
            List<double[]> gradientList = new List<double[]>();

            // 1.3. Initialize update-values:
            List<double[]> updateValues = InitializeNetworkRPropValues(0.1);
            List<double[]> lastUpdateValues = InitializeNetworkRPropValues(0.1);

            // 1.4. Initialize increasing & decreasing constants:
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
                    string handlingErrorText = "";

                    double[] netResult = Network.Handle(InputDatasets[k], ref handlingErrorText);

                    if (netResult == null)
                    {
                        Logger.LogError(ErrorType.NonEqualsInputLengths, handlingErrorText);
                        return;
                    }
                    
                    netLastEpochAnswers.Add(Network.HandleUnsafe(InputDatasets[k]));
                }

                // 2.2. Teaching:
                try
                {
                    // 2.2.1. Calculate epoch error:
                    epochError = RecalculateEpochError(netLastEpochAnswers);

                    gradientList = Network.GetGradients(epochError);

                    // 2.2.2. Calculating update-values:
                    for (int i = 0; i < updateValues.Count; i++)
                    {
                        for (int k = 0; k < updateValues[i].Length; k++)
                        {
                            updateValues[i][k] = CalculateWeightChangeValue(gradientList,
                                                                            lastGradientList,
                                                                            updateValues,
                                                                            lastUpdateValues,
                                                                            i, k,
                                                                            increasingValue,
                                                                            decreasingValue);
                            lastUpdateValues[i][k] = updateValues[i][k];
                        }
                    }

                    lastGradientList = gradientList;

                    // 2.2.3. Teaching net (changing weights):
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

            // 1.2. Initialize gradients list:
            List<double[]> lastGradientList = InitializeNetworkRPropValues(0);
            List<double[]> gradientList = new List<double[]>();

            // 1.3. Initialize update-values:
            List<double[]> updateValues = InitializeNetworkRPropValues(0.1);
            List<double[]> lastUpdateValues = InitializeNetworkRPropValues(0.1);

            // 1.4. Initialize increasing & decreasing constants:
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
                    // 2.2.1. Calculate epoch error:
                    epochError = RecalculateEpochError(netLastEpochAnswers);

                    gradientList = Network.GetGradients(epochError);

                    // 2.2.2. Calculating update-values:
                    for (int i = 0; i < updateValues.Count; i++)
                    {
                        for (int k = 0; k < updateValues[i].Length; k++)
                        {
                            updateValues[i][k] = CalculateWeightChangeValue(gradientList,
                                                                            lastGradientList,
                                                                            updateValues,
                                                                            lastUpdateValues,
                                                                            i, k,
                                                                            increasingValue,
                                                                            decreasingValue);
                            lastUpdateValues[i][k] = updateValues[i][k];
                        }
                    }

                    lastGradientList = gradientList;

                    // 2.2.3. Teaching net (changing weights):
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
            List<double[]> valueList = new List<double[]>();

            // Initialize first values on layers:
            for (int i = 0; i < NetworkStructure.NeuronsByLayers.Length; i++)
            {
                double[] valueRowTemp = new double[NetworkStructure.NeuronsByLayers[i]];

                for (int k = 0; k < NetworkStructure.NeuronsByLayers[i]; k++)
                {
                    valueRowTemp[k] = initializeValues;
                }

                valueList.Add(valueRowTemp);
            }

            return valueList;
        }

        private double RecalculateEpochError(List<double[]> netResultList)
        {
            double sum = 0;

            for (int i = 0; i < netResultList.Count; i++)
            {
                for (int k = 0; k < netResultList[i].Length; k++)
                {
                    double delta = OutputDatasets[i][k] - netResultList[i][k];
                    sum += delta * delta;
                }
            }

            return sum / InputDatasets.Count;
        }

        private double CalculateWeightChangeValue(List<double[]> gradientList,
                                                  List<double[]> lastGradientList,
                                                  List<double[]> updateValues,
                                                  List<double[]> lastUpdateValues,
                                                  int indexI,
                                                  int indexK,
                                                  double increasingValue,
                                                  double decreasingValue)
        {
            double maxDeltaValue = 50;
            double minDeltaValue = 0.000001;

            int gradientChanging = GetSign(gradientList[indexI][indexK] * lastGradientList[indexI][indexK]);

            double weightChange = 0.0;

            if (gradientChanging > 0)
            {
                double deltaValue = updateValues[indexI][indexK] * increasingValue;

                deltaValue = Math.Min(deltaValue, maxDeltaValue);

                updateValues[indexI][indexK] = deltaValue;
                weightChange = GetSign(gradientList[indexI][indexK]) * deltaValue;

                lastGradientList[indexI][indexK] = gradientList[indexI][indexK];
            }

            if (gradientChanging < 0)
            {
                double deltaValue = updateValues[indexI][indexK] * decreasingValue;

                deltaValue = Math.Max(deltaValue, minDeltaValue);

                updateValues[indexI][indexK] = deltaValue;
                weightChange = -lastUpdateValues[indexI][indexK];

                lastGradientList[indexI][indexK] = 0;
            }
            
            if(gradientChanging == 0)
            {
                double deltaValue = updateValues[indexI][indexK];

                weightChange = GetSign(gradientList[indexI][indexK]) * deltaValue;

                lastGradientList[indexI][indexK] = gradientList[indexI][indexK];
            }

            return weightChange;
        }

        /// <summary>
        /// Original method for getting sign of the value
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        private int GetSign(double value)
        {
            if (Math.Abs(value) < 0.000001)
            {
                return 0;
            }

            if (value > 0)
            {
                return 1;
            }

            return -1;
        }

        #endregion
    }
}
