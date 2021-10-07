using NN.Eva.RL.Models;
using System.Collections.Generic;

namespace NN.Eva.RL.Services
{
    public class DatasetGenerator
    {
        /// <summary>
        /// Creating input sets by agents actions tail
        /// </summary>
        /// <param name="tail"></param>
        /// <returns></returns>
        public List<double[]> CreateInputSets(List<RLTail> tail)
        {
            List<double[]> inputSets = new List<double[]>();

            for(int i = 0; i < tail.Count; i++)
            {
                inputSets.Add(tail[i].Environment);
            }

            return inputSets;
        }

        /// <summary>
        /// Creating output sets by agents actions tail
        /// </summary>
        /// <param name="tail"></param>
        /// <returns></returns>
        public List<double[]> CreateOutputSets(List<RLTail> tail)
        {
            List<double[]> outputSets = new List<double[]>();

            for (int i = 0; i < tail.Count; i++)
            {
                outputSets.Add(tail[i].QValues);
            }

            return outputSets;
        }
    }
}
