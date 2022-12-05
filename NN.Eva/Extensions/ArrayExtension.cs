using System.Linq;

namespace NN.Eva.Extensions
{
    public static class ArrayExtension
    {
        /// <summary>
        /// Concatinate second array to first array
        /// </summary>
        /// <param name="baseArray"></param>
        /// <param name="concatedArray"></param>
        /// <returns></returns>
        public static double[] ConcatArray(this double[] baseArray, double[] concatedArray)
        {
            return baseArray.Concat(concatedArray).ToArray();
        }
    }
}
