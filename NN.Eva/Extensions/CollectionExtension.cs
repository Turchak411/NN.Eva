using System.Collections.Generic;

namespace NN.Eva.Extensions
{
    public static class CollectionExtension
    {
        public static List<List<double>> CloneGeneration(this List<List<double>> generation)
        {
            List<List<double>> newGeneration = new List<List<double>>();

            for (int i = 0; i < generation.Count; i++)
            {
                newGeneration.Add(generation[i]);
            }

            return newGeneration;
        }
    }
}
