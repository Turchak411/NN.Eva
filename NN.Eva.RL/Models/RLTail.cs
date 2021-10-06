
namespace NN.Eva.RL.Models
{
    public class RLTail
    {
        /// <summary>
        /// Environment
        /// </summary>
        public double[] Environment { get; set; }

        /// <summary>
        /// Q-values for this state
        /// </summary>
        public double[] QValues { get; set; }

        /// <summary>
        /// Action's index
        /// </summary>
        public int ActionIndex { get; set; }
    }
}
