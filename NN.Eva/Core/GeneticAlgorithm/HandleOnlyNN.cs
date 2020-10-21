using System.Collections.Generic;

namespace NN.Eva.Core.GeneticAlgorithm
{
    public class HandleOnlyNN
    {
        public List<HandleOnlyLayer> LayerList { get; set; } = new List<HandleOnlyLayer>();

        public HandleOnlyNN(List<double> weights)
        {

        }

        /// <summary>
        /// Handling data
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public double[] Handle(double[] data)
        {
            double[] tempData = data;

            for (int i = 0; i < LayerList.Count; i++)
            {
                tempData = LayerList[i].Handle(tempData);
            }

            // There is one double value at the last handle
            return tempData;
        }
    }
}
