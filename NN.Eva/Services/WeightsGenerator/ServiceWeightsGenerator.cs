using System;
using System.Collections.Generic;
using NN.Eva.Models;

namespace NN.Eva.Services.WeightsGenerator
{
    public class ServiceWeightsGenerator
    {
        public void GenerateMemory()
        {
            Generator generator = new Generator();

            Console.Write("Input input-vector length: ");
            int inputVectorLength = Convert.ToInt32(Console.ReadLine());

            int[] netScheme = InputNetScheme();

            generator.GenerateMemory(inputVectorLength, netScheme, "memoryClear.txt");

            Console.WriteLine("Memory generated!");
        }

        public void GenerateMemory(string dataPath)
        {
            Generator generator = new Generator();

            Console.Write("Input input-vector length: ");
            int inputVectorLength = Convert.ToInt32(Console.ReadLine());

            int[] netScheme = InputNetScheme();

            generator.GenerateMemory(inputVectorLength, netScheme, dataPath);

            Console.WriteLine("Memory generated!");
        }

        public void GenerateMemory(string dataPath, int inputVectorLength, int[] netScheme)
        {
            Generator generator = new Generator();
            generator.GenerateMemory(inputVectorLength, netScheme, dataPath);

            Console.WriteLine("Memory generated!");
        }

        public void GenerateDefaultMemory(string dataPath)
        {
            Generator generator = new Generator();

            Console.Write("Input input-vector length: ");
            int inputVectorLength = 3;

            int[] netScheme = new int[2] { 2, 2 };

            generator.GenerateMemory(inputVectorLength, netScheme, dataPath);

            Console.WriteLine("Memory generated!");
        }

        private static int[] InputNetScheme()
        {
            Console.WriteLine("Input net scheme");
            Console.WriteLine("Example for 3 in input layer, 4 in hidden and 2 in out: \"3 4 2\"");
            string[] netSchemeText = Console.ReadLine().Split(' ');
            int[] netScheme = new Int32[netSchemeText.Length];
            for (int i = 0; i < netSchemeText.Length; i++)
            {
                netScheme[i] = Convert.ToInt32(netSchemeText[i]);
            }

            return netScheme;
        }

        public List<double> GenerateMemoryWeights(NetworkStructure networkStructure, Random rnd)
        {
            Generator generator = new Generator();
            return generator.GenerateWeightsVector(networkStructure, rnd);
        }

        public List<double> GenerateEmptyMemoryWeights(NetworkStructure networkStructure)
        {
            Generator generator = new Generator();
            return generator.GenerateEmptyWeightsVector(networkStructure);
        }
    }
}
