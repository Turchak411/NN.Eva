using System.Collections.Generic;

namespace NN.Eva.Models.RL
{
    public class RLConfigModel
    {
        /// <summary>
        /// Positive price for agent's decision
        /// 0.1 prefer to most of the activation functions
        /// </summary>
        public double PositivePrice { get; set; } = 0.1;

        /// <summary>
        /// Negative price for agent's decision
        /// -0.1 prefer to most of the activation functions
        /// </summary>
        public double NegativePrice { get; set; } = -0.1;

        /// <summary>
        /// Action's count
        /// </summary>
        public int ActionsCount { get; set; }

        /// <summary>
        /// Main tail's max length
        /// </summary>
        public int MainTailMaxLength { get; set; }

        /// <summary>
        /// Main training tail (failed actions)
        /// </summary>
        public List<RLTail> MainTail { get; set; } = new List<RLTail>();

        /// <summary>
        /// Fantom tail's max length
        /// </summary>
        public int FantomTailMaxLength { get; set; }

        /// <summary>
        /// Fantom training tail (successed actions)
        /// </summary>
        public List<RLTail> FantomTail { get; set; } = new List<RLTail>();
    }
}
