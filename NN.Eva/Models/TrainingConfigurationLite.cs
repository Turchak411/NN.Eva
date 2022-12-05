
namespace NN.Eva.Models
{
    public class TrainingConfigurationLite
    {
        /// <summary>
        /// Start iteration
        /// </summary>
        public int StartIteration { get; set; } = 0;

        /// <summary>
        /// End iteration
        /// </summary>
        public int EndIteration { get; set; }

        /// <summary>
        /// Memory folder
        /// </summary>
        public string MemoryFolder { get; set; }
    }
}
