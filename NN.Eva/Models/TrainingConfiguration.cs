
namespace NN.Eva.Models
{
    public class TrainingConfiguration
    {
        public TrainingAlgorithmType TrainingAlgorithmType { get; set; }

        public int StartIteration { get; set; }

        public int EndIteration { get; set; }

        public string InputDatasetFilename { get; set; }

        public string OutputDatasetFilename { get; set; }

        public string MemoryFolder { get; set; }
    }
}
