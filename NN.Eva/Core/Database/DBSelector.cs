using System;
using MySql.Data.MySqlClient;
using NN.Eva.Models;
using NN.Eva.Models.Database;
using NN.Eva.Services;
using NN.Eva.Services.Database;

namespace NN.Eva.Core.Database
{
    public class DBSelector
    {
        /// <summary>
        /// Database configuarion
        /// </summary>
        public DatabaseConfig DatabaseConfiguration { get; set; }

        private MySqlConnection _connection;

        private Logger _logger;

        public DBSelector(Logger logger, DatabaseConfig databaseConfiguration)
        {
            _logger = logger;

            DatabaseConfiguration = databaseConfiguration;
        }

        private MySqlConnection GetNewDBConnection()
        {
            MySqlConnection connection = null;

            try
            {
                var dbConfigurator = new DatabaseConfigurator();

                connection = new MySqlConnection(dbConfigurator.ReturnDatabaseConnection(DatabaseConfiguration));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"{DateTime.Now}\nException: {ex}");
                _logger.LogError(ErrorType.DBConnectionError, ex);
            }

            return connection;
        }
    }
}
