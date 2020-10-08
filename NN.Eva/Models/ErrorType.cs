
namespace NN.Eva.Models
{
    public enum ErrorType
    {
        UnknownError,
        MemoryMissing,
        MemoryInitializeError,
        MemoryGenerateError,
        SetMissing,
        TrainError,
        DBConnectionError,
        DBInsertError,
        DBDeleteError,
        DBMemoryLoadError,
        NonEqualsInputLengths,
        OperationWithNonexistentNetwork
    }
}
