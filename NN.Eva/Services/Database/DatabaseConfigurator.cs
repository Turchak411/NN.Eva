using NN.Eva.Models.Database;

namespace NN.Eva.Services.Database
{
    public class DatabaseConfigurator
    {
        public string ReturnDatabaseConnection(DatabaseConfig config)
        {
            return "SERVER=" + config.Server + ";DATABASE=" + config.Database + ";UID=" + config.UID + ";PASSWORD=" + config.Password;
        }
    }
}
