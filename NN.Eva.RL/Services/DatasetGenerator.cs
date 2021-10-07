using System.Linq;
using System.Collections.Generic;
using NN.Eva.Models.RL;

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
                    inputSets.Add(ConcatArrays(tail[i].Environment, actionsVectors));
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

        private double[] ConcatArrays(double[] baseArray, double[] concatedArray)
        {
            return baseArray.Concat(concatedArray).ToArray();
        }
    }
}
