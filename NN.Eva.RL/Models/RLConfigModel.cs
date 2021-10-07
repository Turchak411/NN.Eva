using System.Collections.Generic;

namespace NN.Eva.RL.Models
{
    public class RLConfigModel
    {
        /// <summary>
        /// Positive price for agent's decision
        /// </summary>
        public double PositivePrice { get; set; } = 1;

        /// <summary>
        /// Negative price for agent's decision
        /// </summary>
        public double NegativePrice { get; set; } = -1;

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
