using NN.Eva.Models.RL;
using NN.Eva.Extensions;
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

            for (int i = 0; i < tail.Count; i++)
            {
                for(int k = 0; k < tail[i].QValues.Length; k++)
                {
                    double[] actionsVectors = new double[tail[i].QValues.Length];
                    actionsVectors[k] = 1;
                    inputSets.Add(tail[i].Environment.ConcatArray(actionsVectors));
                }
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
                for (int k = 0; k < tail[i].QValues.Length; k++)
                {
                    outputSets.Add(new double[1] { tail[i].QValues[k] });
                }
            }

            return outputSets;
        }
    }
}
