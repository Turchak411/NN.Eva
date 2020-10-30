using System.Collections.Generic;

namespace NN.Eva.Extensions
{
    public static class CollectionExtension
    {
        /// <summary>
        /// Cloning generation in genetic algorithm
        /// </summary>
        /// <param name="generation"></param>
        /// <returns></returns>
        public static List<List<double>> CloneGeneration(this List<List<double>> generation)
        {
            List<List<double>> newGeneration = new List<List<double>>();

            for (int i = 0; i < generation.Count; i++)
            {
                newGeneration.Add(generation[i]);
            }

            return newGeneration;
        }

        /// <summary>
        /// Cloning chromosome in genetic algorithm
        /// </summary>
        /// <param name="chromosome"></param>
        /// <returns></returns>
        public static List<double> CloneChromosome(this List<double> chromosome)
        {
            List<double> newChromosome = new List<double>();

            for (int i = 0; i < chromosome.Count; i++)
            {
                newChromosome.Add(chromosome[i]);
            }

            return newChromosome;
        }

        /// <summary>
        /// Swapping items in list in genetic algorithm
        /// </summary>
        /// <param name="list"></param>
        /// <param name="index0"></param>
        /// <param name="index1"></param>
        public static void SwapItems(this List<List<double>> list, int index0, int index1)
        {
            List<double> tempItem = list[index0];
            list[index0] = list[index1];
            list[index1] = tempItem;
        }
    }
}
