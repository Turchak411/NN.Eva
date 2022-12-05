
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
        HandleError,
        DBConnectionError,
        DBInsertError,
        DBDeleteError,
        DBMemoryLoadError,
        NonEqualsInputLengths,
        OperationWithNonexistentNetwork,
        NoTrainingConfiguration
    }
}
