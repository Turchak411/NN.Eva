using System.IO;

namespace NN.Eva.Services
{
    public class MemoryChecker
    {
        public bool IsValid(string memoryPath)
        {
            try
            {
                using (StreamReader fileReader = new StreamReader(memoryPath))
                {
                    while (!fileReader.EndOfStream)
                    {
                        string[] readedLine = fileReader.ReadLine().Split(' ');

                        if (readedLine.Length < 3)
                        {
                            return false;
                        }
                    }
                }
            }
            catch
            {
                return false;
            }

            return true;
        }
    }
}
