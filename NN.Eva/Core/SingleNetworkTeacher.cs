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
                case TrainingAlgorithmType.ParallelRProp:
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
                    TrainingRProp();
                    break;
                case TrainingAlgorithmType.ParallelRProp:
                    ParallelTrainingRProp();
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
                    TrainingRProp();
                    break;
                case TrainingAlgorithmType.ParallelRProp:
                    ParallelTrainingRProp();
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

        private void TrainingRProp()
        {
            var neuralNetworkRProp = new NeuralNetworkRProp(new ActivationSigmoid(), NetworkStructure);

            var inputSets = InputDatasets.ToArray();
            var outputSets = OutputDatasets.ToArray();

            double iteration = 0.0;
            while (iteration < TrainingConfiguration.EndIteration - TrainingConfiguration.StartIteration)
            {
                var error = neuralNetworkRProp.Train(inputSets, outputSets);
                Console.WriteLine($"Iteration: {iteration}\t Error: {error}");
                iteration++;
            }

            neuralNetworkRProp.SaveMemory(FileManager.MemoryFolderPath + "\\memory.txt", NetworkStructure);

            // Запись события об успешном обучении:
            LastTrainingSuccess = true;
        }


        private void ParallelTrainingRProp()
        {
            var neuralNetworkRProp = new MultiThreadNeuralNetworkRProp(new ActivationSigmoid {Alpha = 1}, NetworkStructure);

            var inputSets = InputDatasets.ToArray();
            var outputSets = OutputDatasets.ToArray();

            double count = 0;
            double iteration = 0.0;
            while (iteration < TrainingConfiguration.EndIteration - TrainingConfiguration.StartIteration)
            {
                var error = neuralNetworkRProp.Train(inputSets, outputSets);
                if (count >= 500)
                {
                    count = 0;
                    Console.WriteLine($"Iteration: {iteration}\t Error: {error}");
                }

                count++;
                iteration++;
            }

            neuralNetworkRProp.SaveMemory(FileManager.MemoryFolderPath + "\\memory.txt", NetworkStructure);

            // Запись события об успешном обучении:
            LastTrainingSuccess = true;
        }

        #endregion

        #region Genetic algorithm

        private void TrainingGeneticAlg(bool unsafeMode = false)
        {
            GeneticAlgorithmTeacher geneticAlgTeacher = new GeneticAlgorithmTeacher
            {
                NetworkStructure = NetworkStructure,
                InputDatasets = InputDatasets,
                OutputDatasets = OutputDatasets
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
