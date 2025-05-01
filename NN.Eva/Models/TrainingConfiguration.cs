using System;

namespace NN.Eva.Models
{
    public class TrainingConfiguration : TrainingConfigurationLite
    {
        /// <summary>
        /// Training algorithm type
        /// </summary>
        public TrainingAlgorithmType TrainingAlgorithmType { get; set; }

        /// <summary>
        /// Filepath to input dataset
        /// </summary>
        public string InputDatasetFilename { get; set; }

        /// <summary>
        /// Filepath to output dataset
        /// </summary>
        public string OutputDatasetFilename { get; set; }

        /// <summary>
        /// Training set size percent
        /// Default - 0%
        /// </summary>
        public int ValidationSetSize
        {
            get
            {
                return validationSetSize;
            }
            set
            {
                if (value < 0 || value > 100)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "Value can't be below 0 or above 100!");
                }

                validationSetSize = value;
            }
        }

        private int validationSetSize = 0;
    }
}
