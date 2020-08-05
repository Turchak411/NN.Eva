
namespace NN.Eva.Models
{
    public struct TrainObject
    {
        public string _content;
        public double[] _vectorValues;

        public TrainObject(string content, double[] vectorValues)
        {
            _content = content;
            _vectorValues = vectorValues;
        }
    }
}
