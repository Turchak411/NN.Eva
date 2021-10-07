
namespace NN.Eva.Models.RL
{
    public class RLWorkingModel
    {
        /// <summary>
        /// Current environment
        /// </summary>
        public double[] CurrentEnvironment { get; set; }

        /// <summary>
        /// Environment when agent get failed
        /// </summary>
        public double[] FailureEnvironment { get; set; }
    }
}
