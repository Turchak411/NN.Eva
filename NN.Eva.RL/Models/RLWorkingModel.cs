using System.Collections.Generic;

namespace NN.Eva.RL.Models
{
    public class RLWorkingModel
    {
        /// <summary>
        /// Positive price for agent
        /// </summary>
        public double PositivePrice { get; set; } = 1;

        /// <summary>
        /// Negative price for agent
        /// </summary>
        public double NegativePrice { get; set; } = -1;

        /// <summary>
        /// Current environment
        /// </summary>
        public double[] CurrentEnvironment { get; set; }

        /// <summary>
        /// Environment when agent get failed
        /// </summary>
        public double[] FailureEnvironment { get; set; }

        /// <summary>
        /// Action's count
        /// </summary>
        public int ActionsCount { get; set; }

        /// <summary>
        /// Main training tail (failed actions)
        /// </summary>
        public List<RLTail> MainTail { get; set; } = new List<RLTail>();

        /// <summary>
        /// Fantom training tail (successed actions)
        /// </summary>
        public List<RLTail> FantomTail { get; set; } = new List<RLTail>();
    }
}
