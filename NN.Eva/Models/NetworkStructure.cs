using NN.Eva.Models.RL;

namespace NN.Eva.Models
{
    public class NetworkStructure
    {
        public int InputVectorLength { get; set; }

        public int[] NeuronsByLayers { get; set; }

        public double Alpha { get; set; } = 1.0;

        public NetworkStructure() { }

        public NetworkStructure(RLConfigModel rlConfigModel, int inputDataVectorLength)
        {
            InputVectorLength = rlConfigModel.ActionsCount + inputDataVectorLength;
        }

        public NetworkStructure(RLConfigModelTraining rlConfigModel, int inputDataVectorLength)
        {
            InputVectorLength = rlConfigModel.ActionsCount + inputDataVectorLength;
        }
    }
}
