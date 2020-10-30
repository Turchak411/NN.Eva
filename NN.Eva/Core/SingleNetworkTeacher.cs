using System;
using System.Collections.Generic;
using NN.Eva.Models;
using NN.Eva.Services;
using NN.Eva.Core.GeneticAlgorithm;
using NN.Eva.Core.ResilientPropagation;
using NN.Eva.Core.ResilientPropagation.ActivationFunctions;

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

            // Проведение завершающих операций после обучения модели:
            switch (TrainingConfiguration.TrainingAlgorithmType)
            {
                // В случае обучения по генетическому алгоритму - поскольку он сохраняет память сам - бездействие
                case TrainingAlgorithmType.GeneticAlg:
                case TrainingAlgorithmType.RProp:
                    break;
                // В общем случае - сохранение памяти сети:
                case TrainingAlgorithmType.BProp:
                default:
                    Network.SaveMemory(TrainingConfiguration.MemoryFolder + "//memory.txt", NetworkStructure);
                    break;
            }
        }

        private void StartSavedTraining(TrainingAlgorithmType trainingAlgorithm)
        {
            switch (trainingAlgorithm)
            {
                case TrainingAlgorithmType.RProp:
                    SavedTrainingRProp();
                    break;
                case TrainingAlgorithmType.GeneticAlg:
                    TrainingGeneticAlg(false);
                    break;
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
                case TrainingAlgorithmType.RProp:
                    UnsafeTrainingRProp();
                    break;
                case TrainingAlgorithmType.GeneticAlg:
                    TrainingGeneticAlg(true);
                    break;
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
           
        }

        private void UnsafeTrainingRProp()
        {
            var neuralNetworkRProp = new NeuralNetworkRProp(new ActivationSigmoid(), NetworkStructure);

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

        private List<double[]> RecalculateGradientList(List<double[]> gradientList,
                                                       List<double[]> netErrorsList,
                                                       List<double[]> netAnswersList,
                                                       double[] inputSet)
        {
            List<double[]> sumList = new List<double[]>();

            // Sum gradients for input layer:
            double[] sumRowFirstLayer = new double[gradientList[0].Length];

            for (int k = 0; k < gradientList[0].Length; k++)
            {
                sumRowFirstLayer[k] = gradientList[0][k] + netErrorsList[0][k];
            }

            sumList.Add(sumRowFirstLayer);

            // Sum gradients for other layers: 
            for (int i = 1; i < gradientList.Count; i++)
            {
                double[] sumRow = new double[gradientList[i].Length];

                for (int k = 0; k < gradientList[i].Length; k++)
                {
                    sumRow[k] = gradientList[i][k] + netErrorsList[i][k];
                }

                sumList.Add(sumRow);
            }

            // Multipling:
            //for (int k = 0; k < sumList[0].Length; k++)
            //{
            //    sumList[0][k] = sumList[0][k] * inputSet[k];
            //}

            for (int i = 1; i < sumList.Count; i++)
            {
                for (int k = 0; k < sumList[i].Length; k++)
                {
                    sumList[i][k] = sumList[i][k] * netAnswersList[i][k];
                }
            }

            return sumList;
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
            else if (gradientChanging < 0)
            {
                double deltaValue = updateValues[indexI][indexK] * decreasingValue;

                deltaValue = Math.Max(deltaValue, minDeltaValue);

                updateValues[indexI][indexK] = deltaValue;
                weightChange = -lastUpdateValues[indexI][indexK];

                lastGradientList[indexI][indexK] = 0;
            }
            else if (gradientChanging == 0)
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
            if (Math.Abs(value) < 0.00000000000000001)
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

        #region Genetic algorithm

        private void TrainingGeneticAlg(bool unsafeMode = false)
        {
            GeneticAlgorithmTeacher geneticAlgTeacher = new GeneticAlgorithmTeacher
            {
                NetworkStructure = NetworkStructure,
                InputDatasets = InputDatasets,
                OutputDatasets = OutputDatasets,
                Logger = Logger
            };

            geneticAlgTeacher.StartTraining(TrainingConfiguration.EndIteration - TrainingConfiguration.StartIteration,
                                            unsafeMode,
                                            TrainingConfiguration.MemoryFolder + "//memory.txt");

            // Запись события об успешном обучении:
            LastTrainingSuccess = true;
        }

        #endregion
    }
}
