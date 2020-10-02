using System;

namespace NN.Eva.Exceptions
{
    public class NonEqualsInputLengthsException : Exception
    {
        public NonEqualsInputLengthsException(string message) : base(message) { }
    }
}
