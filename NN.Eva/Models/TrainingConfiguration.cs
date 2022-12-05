
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
    }
}
