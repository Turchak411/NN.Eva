using System.Collections.Generic;

namespace NN.Eva.RL.Models
{
    public class RLTrainingModel
    {
        /// <summary>
        /// Memory folder
        /// </summary>
        public string MemoryFolder { get; set; }

        /// <summary>
        /// Memory full path
        /// </summary>
        public string MemoryFullPath { get; set; }

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
