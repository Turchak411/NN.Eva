
namespace NN.Eva.Models
{
    public class NetworkStructure
    {
        public int InputVectorLength { get; set; }

        public int[] NeuronsByLayers { get; set; }

        public double Alpha { get; set; } = 1.0;
    }
}
